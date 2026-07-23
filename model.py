from torch import nn, cat
from transformers import HubertModel
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

class Speech2BlendshapeNetwork(nn.Module, PyTorchModelHubMixin):
    def __init__(self, output_dim, gru_latent_dim, gru_layers, output_fps, gru_dropout, subj_emb_len, emo_emb_len, frozen_encoder_layers, norm, dropout, subj_dropout, embedding_integration, train_subjects):
        super().__init__()
        self.audio_encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")#, local_files_only=True , attn_implementation='flash_attention_2'
        self.audio_dim = self.audio_encoder.encoder.config.hidden_size #768
        self.audio_encoder.feature_extractor._freeze_parameters()
        for name, param in self.audio_encoder.named_parameters():
            if name.startswith('feature_projection'):
                param.requires_grad = False
            if name.startswith("encoder.layers"):
                layer = int(name.split(".")[2])
                if layer in frozen_encoder_layers:
                    param.requires_grad = False

        self.output_dim = output_dim
        self.output_fps = output_fps
        self.input_fps = 50 # for hubert
        self.gru = nn.GRU(self.audio_dim, 
                          gru_latent_dim, 
                          num_layers=gru_layers,
                          batch_first=True,
                          dropout=gru_dropout
        )
        match norm.lower():
            case 'layer':
                self.norm = nn.LayerNorm(gru_latent_dim)
            case 'batch':
                self.norm = nn.BatchNorm1d(gru_latent_dim)
            case _:
                print(f'unknown norm provided {norm}. Use layer or batch')
                self.norm = None

        assert embedding_integration in ['mul', 'add', 'cat']
        self.embedding_integration = embedding_integration
        embedding_size = gru_latent_dim
        if self.embedding_integration == 'cat':
            embedding_size = 5

        self.subj_emb_len = subj_emb_len
        self.train_subjects = train_subjects
        self.subject_emb_layer = None
        if subj_emb_len > 0:
            # Subject embedding
            self.subj_dropout = nn.Dropout(p=subj_dropout)
            self.subject_emb_layer = nn.Linear(subj_emb_len, embedding_size, bias=True)

        self.emo_emb_layer = None
        self.emo_emb_len = emo_emb_len
        if emo_emb_len > 0:
            # emotion embedding
            self.emo_emb_layer = nn.Linear(emo_emb_len, embedding_size, bias=True)

        self.d = nn.Dropout(p=dropout)
        
        final_dim = gru_latent_dim
        if self.embedding_integration == 'cat':
            final_dim = gru_latent_dim + 2*embedding_size
        self.final = nn.Linear(final_dim, self.output_dim)


    
    def resize(self, hidden_state):
        new_size = round(hidden_state.size()[1] / self.input_fps) * self.output_fps
        #print(f'hidden_state.size()[1]: {hidden_state.size()[1]}, self.input_fps: {self.input_fps}, divided: {hidden_state.size()[1] // self.input_fps}, output_fps: {self.output_fps} new_size {new_size}')
        hidden_state = hidden_state.permute(0, 2, 1)
        hidden_state = F.interpolate(hidden_state, size=new_size)
        hidden_state = hidden_state.permute(0, 2, 1)
        return hidden_state


    def forward(self, x, subj, emo):
        hidden_state = self.audio_encoder(x).last_hidden_state
        hidden_state = self.resize(hidden_state)
        #h0 = torch.zeros(self.gru_layer_dim, hidden_state.size()[0], self.gru_latent_dim).requires_grad_().cuda()
        #output, _ = self.gru(hidden_state, h0)
        output, _ = self.gru(hidden_state)
        if self.norm is not None:
            output = self.norm(output)
        if self.subject_emb_layer is not None:
            subj = self.subj_dropout(subj)
            subj_embedding = self.subject_emb_layer(subj)
            if self.embedding_integration == 'mul':
                output = output * subj_embedding.unsqueeze(1)
            elif self.embedding_integration == 'add':
                output = output + subj_embedding.unsqueeze(1)
            else:
                subj_embedding = subj_embedding.unsqueeze(1).repeat((1, output.size()[1], 1))
                output = cat((output, subj_embedding), 2)
        if self.emo_emb_layer is not None:
            emo_embedding = self.emo_emb_layer(emo)
            if self.embedding_integration == 'mul':
                output = output * emo_embedding.unsqueeze(1)
            elif self.embedding_integration == 'add':
                output = output + emo_embedding.unsqueeze(1)
            else:
                emo_embedding = emo_embedding.unsqueeze(1).repeat((1, output.size()[1], 1))
                output = cat((output, emo_embedding), 2)
        output = self.d(output)
        output = self.final(output)
        return output


if __name__ == "__main__":
    model = Speech2BlendshapeNetwork()
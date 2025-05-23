import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, act='tanh', drop_rate=0.1, bn_first=True):
        super(MLP, self).__init__()
        layers = []
        if bn_first:
            layers.append(nn.BatchNorm1d(input_dim))
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.LeakyReLU(0.01) if act == 'relu' else nn.Tanh())
        layers.append(nn.Dropout(drop_rate))

        for i in range(1, len(hidden_dims)):
            layers.append(nn.BatchNorm1d(hidden_dims[i - 1]))
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.LeakyReLU(0.01) if act == 'relu' else nn.Tanh())
            layers.append(nn.Dropout(drop_rate))

        self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.mlp(x)


class Discriminator(nn.Module):
    def __init__(self, emb_dim, hidden_dims, act='tanh', drop_rate=0.5):
        super(Discriminator, self).__init__()
        self.user_mlp = MLP(emb_dim, hidden_dims, act, drop_rate, bn_first=True)
        self.item_mlp = MLP(emb_dim, hidden_dims, act, drop_rate, bn_first=True)

    def forward(self, uemb, iemb):
        u_proj = self.user_mlp(uemb)
        i_proj = self.item_mlp(iemb)
        return torch.sum(u_proj * i_proj, dim=-1)


class GAR(nn.Module):
    def __init__(self, emb_dim, content_dim, g_layer=[200, 200], d_layer=[200, 200],
                 g_act='tanh', d_act='tanh', g_drop=0.1, d_drop=0.5, alpha=0.05, beta=0.1):
        super(GAR, self).__init__()
        self.emb_dim = 64  # fixed embedding dimension
        self.content_dim = content_dim
        self.alpha = alpha
        self.beta = beta

        # Generator
        self.generator = MLP(content_dim, g_layer, g_act, g_drop, bn_first=False)
        self.gen_projection = nn.Linear(g_layer[-1], self.emb_dim)
        nn.init.xavier_uniform_(self.gen_projection.weight)
        if self.gen_projection.bias is not None:
            nn.init.constant_(self.gen_projection.bias, 0)

        # Discriminator
        self.discriminator = Discriminator(self.emb_dim, d_layer, d_act, d_drop)

        # Optimizers
        self.g_optimizer = optim.Adam(list(self.generator.parameters()) + list(self.gen_projection.parameters()),
                                      lr=1e-3, weight_decay=1e-3)
        self.d_optimizer = optim.Adam(self.discriminator.parameters(), lr=1e-3, weight_decay=1e-3)

        # Loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward_generator(self, content, training=True):
        gen_out = self.generator(content)
        return self.gen_projection(gen_out)

    def forward_discriminator(self, uemb, iemb, training=True):
        return self.discriminator(uemb, iemb)

    def train_step(self, content, real_emb, neg_emb, opp_emb,args):
        batch_size = content.size(0)

        # Train Discriminator
        self.d_optimizer.zero_grad()
        gen_emb = self.forward_generator(content, training=True)

        uemb = opp_emb.repeat(3, 1)
        iemb = torch.cat([real_emb, neg_emb, gen_emb], dim=0)
        d_out = self.forward_discriminator(uemb, iemb, training=True)
        d_out = d_out.view(3, -1).t()

        real_logit, neg_logit, fake_logit = d_out[:, 0], d_out[:, 1], d_out[:, 2]
        target = torch.ones_like(real_logit)
        d_loss = self.bce_loss(real_logit - (1 - self.beta) * fake_logit - self.beta * neg_logit, target)
        d_loss.backward()
        self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        gen_emb = self.forward_generator(content, training=True)
        g_out = self.forward_discriminator(opp_emb, gen_emb, training=True)
        d_out = self.forward_discriminator(opp_emb, real_emb, training=True)

        g_adv_loss = self.bce_loss(g_out - d_out, torch.ones_like(g_out))
        sim_loss = torch.mean(torch.abs(gen_emb - real_emb))
        g_loss = (1.0 - self.alpha) * g_adv_loss + self.alpha * sim_loss
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item(), sim_loss.item()

    def get_item_emb(self, content, item_emb, warm_item, cold_item):
        """Returns updated item embeddings after passing cold items through the generator and MLP."""
        item_emb = item_emb.clone()
        item_emb[cold_item] = self.forward_generator(content[cold_item], training=False)
        with torch.no_grad():
            item_emb = self.discriminator.item_mlp(item_emb)
        return item_emb

    def get_user_emb(self, user_emb):
        with torch.no_grad():
            return self.discriminator.user_mlp(user_emb)

    def get_user_rating(self, uemb, iemb):
        return torch.matmul(uemb, iemb.t())

    def get_ranked_rating(self, ratings, k):
        """Return top-k scores and indices for the given ratings."""
        with torch.no_grad():
            _, indices = torch.topk(ratings, k, dim=1)
            return ratings[:, :k], indices.cpu().numpy()

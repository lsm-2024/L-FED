import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Decoder, DecoderLayer, Encoder, EncoderLayer, ConvLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding
from config import CONFIG

class TransModel(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs):
        super(nn.Module, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout)
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout, output_attention=False),
                        configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.d_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):

        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]
def train_model(model, train_loader, val_loader, test_loader, scaler):
    """模型训练、验证与测试"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["learning_rate"])

    # 训练记录
    train_loss = []
    val_loss = []
    test_loss = []

    for epoch in range(CONFIG["num_epochs"]):
        # 训练阶段
        model.train()
        epoch_train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(CONFIG["device"])
            batch_y = batch_y.to(CONFIG["device"])

            # 前向传播
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            epoch_train_loss += loss.item()

        # 验证阶段
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(CONFIG["device"])
                val_y = val_y.to(CONFIG["device"])
                preds = model(val_x)
                epoch_val_loss += criterion(preds, val_y).item()

        # 测试阶段（每个epoch都查看测试集表现，但不参与参数更新）
        epoch_test_loss = 0
        with torch.no_grad():
            for test_x, test_y in test_loader:
                test_x = test_x.to(CONFIG["device"])
                test_y = test_y.to(CONFIG["device"])
                preds = model(test_x)
                epoch_test_loss += criterion(preds, test_y).item()

        # 记录损失
        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_test_loss = epoch_test_loss / len(test_loader)

        train_loss.append(avg_train_loss)
        val_loss.append(avg_val_loss)
        test_loss.append(avg_test_loss)

        # 打印进度
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{CONFIG['num_epochs']}] | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f} | "
                  f"Test Loss: {avg_test_loss:.4f}")

    # 绘制损失曲线
    #plt.figure(figsize=(10, 5))
    #plt.plot(train_loss, label="Training Loss")
    #plt.plot(val_loss, label="Validation Loss")
    #plt.plot(test_loss, label="Test Loss")
    #plt.title("Loss Evolution")
    #plt.xlabel("Epochs")
    #plt.ylabel("MSE Loss")
    #plt.legend()
    #plt.show()

    return model
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# 1. Veri Setini Yükle (Cora Veriseti)
# Bu işlem veriyi otomatik olarak 'data/Cora' klasörüne indirecektir.
dataset = Planetoid(root='data/Cora', name='Cora')

print(f'Veriseti: {dataset}:')
print('======================')
print(f'Grafik sayısı: {len(dataset)}')
print(f'Özellik sayısı (Feature): {dataset.num_features}')
print(f'Sınıf sayısı (Classes): {dataset.num_classes}')

# Verisetindeki ilk (ve tek) grafiği alıyoruz
data = dataset[0] 

# 2. GNN Modelini Tanımla
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # İlk GCN katmanı: Girdi özellikleri -> Gizli katman (16 nöron)
        self.conv1 = GCNConv(dataset.num_features, 16)
        # İkinci GCN katmanı: Gizli katman -> Çıktı sınıfları (7 kategori)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # Katman 1
        x = self.conv1(x, edge_index)
        x = F.relu(x) # Aktivasyon fonksiyonu
        x = F.dropout(x, training=self.training) # Aşırı öğrenmeyi önleme (Dropout)

        # Katman 2
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

# Modeli ve optimizasyon aracını hazırla
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# 3. Eğitim Döngüsü
print("\nEğitim Başlıyor...")
model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data)
    # Sadece eğitim maskesi olan düğümler için hatayı hesapla
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    
    if epoch % 20 == 0:
        print(f'Epoch {epoch} | Loss: {loss.item():.4f}')

# 4. Modeli Test Et
model.eval()
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'\nTest Doğruluğu: {acc:.4f}')
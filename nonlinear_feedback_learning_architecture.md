# Non-Lineer, Geri Bildirimli, Öğrenen Çözüm Mimarisi

Bu doküman, mevcut omurgayı bozmadan (özellikle `SemanticSignalModule`, `MathASTBuilder`, solver seçici, doğrulayıcılar, DAG yürütücüler, `TemporalStateMemory`, `EntropyThresholdManager`, `CompletenessChecker`, opsiyonel `ThinkingLoopBackpropEngine`) **katmanlı ve sürekli öğrenen** bir mimariyi tek dosyada tanımlar.

---

## 1) Mevcut Mimari Akış Şeması (As-Is)

```text
[INPUT SORU]
    ↓
[SemanticSignalModule]
    ↓
[MathASTBuilder / Problem Type Router]
    ↓
[Dependency Graph Builder]
    ↓
[ConstraintGraphBuilder]
    ├─ kural yayılımı
    ├─ çelişki budama
    └─ alan kısıtları
    ↓
[OperationOrderResolver]
    ├─ precedence graph
    ├─ non-commutative sıralama
    └─ yürütme planı
    ↓
[SymbolicConsistencyEngine]
    ├─ sembol eşleme
    ├─ değişken tutarlılığı
    └─ anlam kayması kontrolü
    ↓
[StepVerificationEngine]
    ├─ her adım mini-proof
    ├─ ara sonuç doğrulama
    └─ step-level rollback
    ↓
[EntropyThresholdManager]
    ├─ belirsizlik ölçer
    ├─ CoT derinleştirir
    └─ gerekirse soru sorar
    ↓
[Solver Selection Engine]
    ├─ BayesSolver
    ├─ MarkovSolver
    ├─ GameTheorySolver
    ├─ DifferentialDynamicsSolver
    └─ Neural/DL Solver
    ↓
[Multi-Solver Execution]
    ├─ symbolic branch
    ├─ numeric branch
    ├─ simulation branch
    └─ neural branch
    ↓
[Multi-Solver Consensus Engine]
    ├─ majority vote
    ├─ confidence weighting
    └─ disagreement analysis
    ↓
[Causal Inference Engine]
    ├─ korelasyon ≠ nedensellik ayrımı
    ├─ counterfactual check
    └─ root-cause tagging
    ↓
[Self-Correction Loop]
    ├─ validator feedback
    ├─ recovery / rollback
    └─ graph rewriting
    ↓
[Neural Meta-Learner]
    ├─ hangi solver?
    ├─ hangi sıra?
    └─ hangi derinlik?
    ↓
[Experience Replay Memory (PKL)]
    ├─ (state, action, reward)
    ├─ başarılı çözüm örnekleri
    └─ hata örnekleri
    ↓
[Graph Neural Network Layer]
    ├─ DAG embedding
    ├─ structural encoding
    └─ pattern memory
    ↓
[Energy-Based Model]
    ├─ düşük enerji = daha tutarlı çözüm
    ├─ çözüm aday kıyaslama
    └─ belirsizlik cezalandırma
    ↓
[TemporalStateMemory / Long-Horizon Memory]
    ├─ zaman çürümesi
    ├─ bağlam izi
    └─ soru-tipleri arası transfer
    ↓
[FINAL OUTPUT]
```

---

## 2) Hedef Davranış: Non-Lineer + Öz-Değerlendirmeli + Sürekli Öğrenen Döngüler

Bu mimari doğrusal boru hattı olarak değil, birbiriyle konuşan döngüler olarak çalışır:

```text
A. Anlama Döngüsü
[input] → [semantic parse] → [AST] → [constraint graph] → [tutarlılık]

B. Çözüm Döngüsü
[solver seçimi] → [çoklu solver] → [consensus] → [cevap adayı]

C. Denetim Döngüsü
[step/final validation] → [hata tespiti] → [repair/rollback] → [yeniden çözüm]

D. Öğrenme Döngüsü
[sonuç] → [reward shaping] → [replay] → [policy update] → [bir sonraki soruda iyileşme]

E. Bellek Döngüsü
[embedding] → [temporal memory] → [pattern reuse] → [transfer]
```

---

## 3) Modül Envanteri: Mevcut / Kısmi / Tamamlanacak

| Modül | Durum | Mevcut Koddan Kullanım | Tamamlama Notu |
|---|---|---|---|
| SemanticSignalModule | Mevcut | Giriş anlamlandırma | Sinyal kalibrasyonu + uncertainty label eklenecek |
| MathASTBuilder | Mevcut | Problem tipleme ve AST | Alan-spesifik constraint üretimi artırılacak |
| DependencyGraph/DAG yürütücüler | Mevcut | Topo sıra, node planı | Graph rewriting geri besleme ile güçlendirilecek |
| ValidationRouter + doğrulayıcılar | Mevcut | Adım/final denetim | Step-level kanıt izine standardizasyon |
| EntropyThresholdManager | Mevcut | Belirsizlikte derinleşme | Calibrated uncertainty ile birleştirilecek |
| TemporalStateMemory | Mevcut | Zaman çürümesi + durum | Uzun ufuk transfer hafızası ayrı namespace |
| CompletenessChecker | Mevcut | Kapanış/tamlık kontrolü | Counterfactual tamlık kontrolü eklenecek |
| ThinkingLoopBackpropEngine | Opsiyonel mevcut | Düşünme döngüsü geri besleme | Reward shaping ile ortak loss bağlanacak |
| EmbeddingGenerator + ChromaDB + RAGRetriever | Mevcut | Embedding tabanlı geri çağırma | Replay ile çift yönlü bağ kurulacak |
| Q-learning benzeri politika | Mevcut | Strateji/route seçimi | Policy-Value ayrımıyla neural meta-öğrenmeye geçirilecek |

---

## 4) Eksik Modüller ve Entegrasyon Tasarımı

Aşağıdaki modüller yeni ek bileşenler olarak tasarlanır; ancak mevcut modülleri **değiştirmek yerine sarmalayarak (adapter/facade)** ilerler.

1. **ConstraintGraphBuilder**
   - Girdi: AST + domain varsayımları
   - Çıktı: kısıt DAG’ı (hard/soft constraint)
   - Entegrasyon: `MathASTBuilder` çıkışından sonra çalışır.

2. **OperationOrderResolver**
   - Girdi: dependency graph + non-commutative işaretler
   - Çıktı: icra planı + alternatif planlar
   - Entegrasyon: DAG executor öncesi planlayıcı.

3. **SymbolicConsistencyEngine**
   - Girdi: sembol tablosu + adımlar
   - Çıktı: tutarlılık raporu + düzeltme önerileri
   - Entegrasyon: ValidationRouter ile çift yönlü.

4. **StepVerificationEngine**
   - Girdi: adım listesi + ara sonuçlar
   - Çıktı: mini-proof doğrulama, rollback noktası
   - Entegrasyon: CompletenessChecker öncesi zorunlu geçit.

5. **MultiSolverConsensusEngine**
   - Girdi: solver branch çıktıları
   - Çıktı: ağırlıklı uzlaşma + çelişki matrisi
   - Entegrasyon: solver orkestrasyon katmanı.

6. **NeuralMetaLearner (Policy+Value)**
   - Girdi: temsil vektörü + tarihsel performans
   - Çıktı: solver sırası, derinlik, beklenen fayda
   - Entegrasyon: mevcut Q-policy üzerine üst katman.

7. **ExperienceReplayMemory (PKL)**
   - Girdi: `(state, action, reward, next_state, done)`
   - Çıktı: eğitim batch’leri
   - Entegrasyon: ThinkingLoopBackprop + Q-learning update.

8. **GraphNeuralLayer**
   - Girdi: çözüm DAG’ı
   - Çıktı: yapısal embedding
   - Entegrasyon: DeepRepresentationEncoder ile concat.

9. **CausalInferenceEngine**
   - Girdi: çözüm izleri, müdahale senaryoları
   - Çıktı: nedensel etiketler + root-cause
   - Entegrasyon: consensus sonrası güven filtresi.

10. **EnergyBasedScorer**
    - Girdi: çözüm adayları
    - Çıktı: enerji skoru (düşük daha iyi)
    - Entegrasyon: final selection katmanı.

11. **LongHorizonTemporalMemory**
    - Girdi: soru tipi, başarı profili, zaman damgası
    - Çıktı: transfer prior
    - Entegrasyon: TemporalStateMemory ile iki seviye bellek.

12. **PolicyStrategyRouter**
    - Girdi: semantik sinyal + value tahmini
    - Çıktı: sembolik/sayısal/simülasyon/nöral rota
    - Entegrasyon: mevcut router’ı geriye uyumlu genişletir.

13. **UncertaintyCalibrationModule**
    - Girdi: ham confidence
    - Çıktı: kalibre edilmiş güven skoru
    - Entegrasyon: EntropyThresholdManager’a sinyal sağlar.

14. **CounterfactualChecker**
    - Girdi: kritik varsayım seti
    - Çıktı: “varsayım değişirse ne olur” raporu
    - Entegrasyon: final cevap öncesi sağlamlık testi.

15. **RecoveryRollbackManager**
    - Girdi: hata tipi + snapshot
    - Çıktı: güvenli geri sarma / branch switch
    - Entegrasyon: mevcut recovery engine’in state-aware versiyonu.

16. **RewardShapingModule**
    - Girdi: doğruluk + tutarlılık + açıklanabilirlik + maliyet
    - Çıktı: yoğun reward
    - Entegrasyon: Q-learning ve backprop ortak ödül kaynağı.

17. **DomainSpecificPhysicsAdapter**
    - Girdi: domain etiketi (fizik/oyun kuramı/olasılık...)
    - Çıktı: domain kural seti ve solver konfigürasyonu
    - Entegrasyon: Solver Selection Engine öncesi.

18. **PromptConstraintCompiler**
    - Girdi: ham soru metni
    - Çıktı: normalize kısıt dili + hedef fonksiyon
    - Entegrasyon: SemanticSignalModule ön-kademesi.

---

## 5) Öğrenen Nöral Akış (Her Soruda Kendini Güncelleyen)

```text
Soru
  ↓
Tokenizer / Signal Extractor
  ↓
Embedding
  ↓
GNN or Transformer Encoder
  ↓
Policy Head → solver seçimi
  ↓
Value Head → beklenen fayda / güven
  ↓
Çoklu Solver Çalıştırma
  ↓
Validator + Counterfactual Checker
  ↓
Reward Shaping
  ↓
Replay Buffer (PKL)
  ↓
Weight Update (online veya mini-batch)
  ↓
Bir sonraki soruda güncellenmiş strateji
```

---

## 6) Uygulama Fazları (Kademeli Geçiş Planı)

### Faz-1 (Düşük Risk / Hızlı Kazanç)
- ConstraintGraphBuilder
- StepVerificationEngine
- MultiSolverConsensusEngine
- UncertaintyCalibrationModule

### Faz-2 (Öğrenme ve Dayanıklılık)
- ExperienceReplayMemory
- RewardShapingModule
- RecoveryRollbackManager
- CounterfactualChecker

### Faz-3 (Derin Nöral İyileştirme)
- NeuralMetaLearner (policy/value)
- GraphNeuralLayer
- EnergyBasedScorer
- LongHorizonTemporalMemory

### Faz-4 (Domain Uzmanlaşması)
- DomainSpecificPhysicsAdapter
- PromptConstraintCompiler
- CausalInferenceEngine (domain etiketli)

---

## 7) Başarı Kriterleri (Ölçülebilir)

- **Doğruluk:** final answer başarı oranı
- **Adım Tutarlılığı:** StepVerification geçiş oranı
- **Kalibrasyon:** confidence vs actual başarı farkı (ECE/Brier)
- **Geri Sarma Etkinliği:** rollback sonrası kurtarılan örnek oranı
- **Öğrenme Kazancı:** N görev sonrası performans artışı
- **Transfer Gücü:** yeni soru tipinde soğuk başlangıca göre fark

---

## 8) Kısa Sonuç

Bu tasarım, mevcut kod tabanındaki güçlü iskeleti koruyup modüler eklemelerle sistemi şu profile taşır:
- lineer değil **döngüsel-akıllı**,
- tek çözücü değil **uzlaşmalı çoklu çözücü**,
- statik değil **geri bildirimle öğrenen**,
- kısa hafızalı değil **uzun ufuk transfer yapan**,
- sadece doğru sonuca odaklı değil **adım kalitesi + nedensellik + güven kalibrasyonu** odaklı.

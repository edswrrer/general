# Thinking Loop + Backprop ile Tamamlanmış Mimari

Aşağıdaki akış, "ne düşündüğünü anlayan / ne anladığını düşünen" bir sistemi **öğrenilebilir** hale getirir.

```mermaid
flowchart TD
A[Soru] --> B[Semantic Parser]

B --> C[Latent Thought Encoder (LTE)]
C --> D[Problem Representation]

D --> E[Solver Engine]
E --> F[Solution Steps]

F --> G[Reason Abstraction Layer]
G --> H[Thought Vector]

H --> I[Natural Language Thought Decoder (RNLT)]
I --> J[Explanation Output]

J --> K[Explanation Critic]
K --> L{Yeterince iyi mi?}

L -- Hayır --> M[Rewrite / Improve]
M --> I

L -- Evet --> N[Final Output]

%% LEARNING LOOP
N --> O[Loss Computation]
J --> O
H --> O
F --> O

O --> P[Backpropagation]

P --> C
P --> E
P --> I
P --> K
```

## Eksik 5 kritik bloğun tamamlanmış hali

1. **Gizli Düşünce Kodlayıcı (LTE)**
   - Girdi + problem bağlamını, çözüm niyetini taşıyan bir latent vektöre çevirir.
   - Çıkış: `z_thought` (süreci temsil eden gizli düşünce uzayı).

2. **Düşünce -> Dil Çözücü (RNLT)**
   - `z_thought` ve çözüm adımlarından öğrenilmiş açıklama üretir.
   - Kural tabanlı değil; seq2seq/transformer tabanlı eğitimli decoder.

3. **Açıklama Kaybı (Explainability Loss)**
   - Sadece doğru cevap değil, kaliteli anlatım da optimize edilir.
   - Örnek toplam kayıp:
     - `L_total = λ1*L_answer + λ2*L_explanation + λ3*L_faithfulness + λ4*L_critic`

4. **Öz-yansıtma döngüsü (Self-Reflection / Critic Loop)**
   - Critic; tutarlılık, açıklık, adım uyumu, halüsinasyon riski skorları üretir.
   - Skor düşükse `Rewrite/Improve` tekrar çalışır.

5. **Akıl yürütme yolu üzerinden geri yayılım**
   - Loss yalnız son cevaba değil, ara temsil ve açıklama katmanlarına da akar.
   - Böylece sistem "sonucu bulma" + "nasıl anlattığını öğrenme" birlikte geliştirir.

## Pratik eğitim şeması

- **Aşama 1:** Solver + Parser temel doğruluk (cevap odaklı pretrain)
- **Aşama 2:** LTE + RNLT ile açıklama üretimi (teacher forcing / SFT)
- **Aşama 3:** Critic ile yeniden yazma döngüsü (iterative refinement)
- **Aşama 4:** Çok amaçlı loss ile uçtan uca fine-tune

## Minimum modül arayüzleri (öneri)

- `z_thought = LTE(x, context)`
- `steps = Solver(z_thought, problem_repr)`
- `exp = RNLT(z_thought, steps)`
- `score = Critic(exp, steps, answer)`
- `L_total = MultiObjectiveLoss(answer, exp, score, supervision)`

Bu düzenle sistem:
- çözümü üretir,
- çözümü açıklar,
- açıklamasını eleştirir,
- eleştiriden öğrenir.

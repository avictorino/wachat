# Manual Validation Guide

## Purpose
This guide helps you manually validate that the simulator and real chat produce equivalent responses with proper biblical language in addiction contexts.

## Prerequisites
- Ollama running locally with wachat-v9 model
- Database configured and migrations applied
- Application running

## Test Scenarios

### Scenario 1: Alcohol Struggle (Initial Message)

**Test in Real Chat:**
1. Go to chat interface
2. Create new profile (name: "João")
3. Send message: "Estou bebendo todo dia e não consigo parar"

**Test in Simulator:**
```bash
python manage.py simulate alcool
```

**Expected Results (BOTH modes should match):**
- ✅ Single consolidated message (not multiple fragments)
- ✅ Biblical vocabulary present: tentação, fraqueza, vigilância, arrependimento, graça, restauração
- ✅ God as presence/support (e.g., "Deus permanece", "presença de Deus", "graça de Deus")
- ✅ NO clinical language (e.g., "dependência química", "transtorno")
- ✅ NO self-help clichés (e.g., "você consegue!", "força de vontade")
- ✅ Separates error from identity (e.g., "Isso não te define")
- ✅ Hope rooted in faith, not willpower

**Example Good Response:**
```
Isso não te define. A tentação é real e a fraqueza da carne é parte da condição humana. 
Mesmo quando caímos, a presença de Deus não se afasta.

A vigilância do coração não é sobre perfeição, é sobre cuidado consigo mesmo. 
O arrependimento é voltar-se, não se punir. O que mais está pesando no seu coração agora?
```

---

### Scenario 2: Drug Relapse

**Test in Real Chat:**
1. Send message: "Voltei a usar drogas depois de meses limpo. Me sinto fraco"

**Test in Simulator:**
```bash
python manage.py simulate drogas
```

**Expected Results (BOTH modes should match):**
- ✅ Single message
- ✅ Acknowledges courage to speak
- ✅ Biblical framing: graça, restauração, misericórdia
- ✅ God as companion, not judge
- ✅ Reinforces dignity and worth
- ✅ NO psychological jargon

**Example Good Response:**
```
Falar sobre isso exige coragem. A queda não apaga quem você é diante de Deus. 
A graça está no recomeço, não na ausência de quedas. Mesmo na fraqueza, 
Deus sustenta.

Onde você sente que Deus ainda está perto, mesmo agora?
```

---

### Scenario 3: Multi-Turn Conversation (Context Handling Test)

**Test in Real Chat:**
1. Send: "Não aguento mais tentar e falhar"
2. Wait for bot response
3. Send: "Já tentei parar mil vezes"
4. Wait for bot response
5. Send: "Me sinto inútil"

**Test in Simulator:**
Look at conversation flow in generated transcript

**Expected Results (BOTH modes should match):**
- ✅ Each bot response is a single message
- ✅ Context flows naturally (bot remembers previous exchanges)
- ✅ No repetition of empathy statements
- ✅ Biblical language maintained throughout
- ✅ God's presence reinforced without being preachy
- ✅ No loops (asking same question twice)

---

### Scenario 4: Non-Addiction Context (Control Test)

**Test in Real Chat:**
1. Send: "Estou com medo de uma cirurgia"

**Test in Simulator:**
```bash
python manage.py simulate doenca
```

**Expected Results:**
- ✅ Single message (no fragmentation)
- ✅ Spiritual framing still present but less emphasis on temptation/vigilância
- ✅ Biblical language appropriate to context (presença, cuidado, sustento)
- ✅ Same conversational quality as addiction themes

---

## Validation Checklist

### Message Structure
- [ ] Real chat produces exactly 1 message per bot turn
- [ ] Simulator produces exactly 1 message per bot turn
- [ ] No paragraph fragmentation into multiple messages

### Biblical Vocabulary (Addiction Contexts)
- [ ] "tentação" used naturally (not "vício inevitável")
- [ ] "fraqueza da carne" mentioned (as humanity, not condemnation)
- [ ] "vigilância" or "vigilância do coração" present
- [ ] "arrependimento" used as "voltar-se", not "punir-se"
- [ ] "graça" emphasized (greater than culpa)
- [ ] "restauração" mentioned (being remade)
- [ ] "presença de Deus" in weakness emphasized

### Prohibited Language (Addiction Contexts)
- [ ] NO "dependência química" or clinical terms
- [ ] NO "você consegue!" or "força de vontade" clichés
- [ ] NO purely psychological framing without spiritual dimension
- [ ] NO "Deus está desapontado/distante/punindo"

### God's Presence
- [ ] God as presence and support (not judge)
- [ ] God active, not abstract concept
- [ ] Simple, direct phrases (e.g., "Deus permanece", "Deus sustenta")
- [ ] NO theological explanations or sermons

### Response Quality
- [ ] Tone is warm, calm, human
- [ ] 2-3 paragraphs maximum
- [ ] At most one question (preferably at end)
- [ ] No repetition of user's exact words
- [ ] Validates without explaining too much

### Simulator vs Real Chat Equivalence
- [ ] Same biblical vocabulary density
- [ ] Same spiritual framing tone
- [ ] Same message length and structure
- [ ] Same contextual awareness
- [ ] Indistinguishable by human reader (except for name/timestamp)

---

## Troubleshooting

### If Biblical Language is Missing:
1. Verify Modelfile was updated: `cat model/Modelfile | grep "TEMAS DE LUTA"`
2. Rebuild Ollama model: `ollama create wachat-v9 -f model/Modelfile`
3. Restart application
4. Check theme is active: Profile should have `prompt_theme` set

### If Messages are Still Fragmented:
1. Check code changes applied: `git log --oneline | head -5`
2. Verify `generate_intent_response()` returns single message
3. Check views.py saves all messages correctly

### If Simulator Differs from Real Chat:
1. Verify both use same `generate_intent_response()` call
2. Check conversation context is identical (last 5 messages, excluding current)
3. Verify temperature is same (0.65) for bot responses
4. Check RAG retrieval is working for both

---

## Success Criteria

**Pass**: If after testing 3-5 conversations in both modes:
1. All messages are single, unified responses
2. Biblical vocabulary naturally present in addiction contexts
3. God consistently framed as presence/support
4. No clinical language or self-help clichés
5. Simulator and real chat responses are semantically equivalent
6. A neutral reader cannot distinguish which is which

**Fail**: If any of the above criteria are not met consistently

---

## Reporting Issues

If validation fails:
1. Document which scenario failed
2. Copy exact user input and bot response
3. Note which mode (simulator vs real chat)
4. Describe what was wrong (missing biblical language, fragmented, etc.)
5. Check logs for any errors

## Notes

- Biblical language should be **natural**, not forced
- God should be **implicit presence**, not sermonic
- Responses should be **concise** (2-3 paragraphs max)
- Single reflective question at most, never multiple questions
- Both modes must produce **identical quality** responses

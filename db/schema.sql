-- База данных проекта «Томление: толкования NSM языковыми моделями»
-- В публичной версии (kmiha_project_public.db) таблица corpus заменена на corpus_meta
-- (example_id, source, form, domain_auto, text_len) — без текстов контекстов.
CREATE TABLE corpus (            -- весь корпус употреблений
  example_id TEXT PRIMARY KEY, source TEXT, form TEXT,
  left_ctx TEXT, right_ctx TEXT, full_text TEXT,
  domain_auto TEXT              -- грубая разметка по ключевым словам: religious / political / erotic / psych / NULL
);
CREATE TABLE prompt_contexts (   -- 24 контекста, вошедшие в промпт
  position INTEGER PRIMARY KEY, example_id TEXT REFERENCES corpus(example_id),
  domain TEXT, text_in_prompt TEXT
);
CREATE TABLE primes (            -- 65 примитивов на трёх языках
  id INTEGER PRIMARY KEY, category_en TEXT, category_ru TEXT, en TEXT, ru TEXT, he TEXT
);
CREATE TABLE prompts (lang TEXT PRIMARY KEY, template TEXT);  -- {CONTEXTS} подставляется из prompt_contexts
CREATE TABLE models (name TEXT PRIMARY KEY, openrouter_id TEXT, developer TEXT, country TEXT, status TEXT, note TEXT);
CREATE TABLE runs (              -- 30 ответов моделей
  code TEXT PRIMARY KEY,         -- слепой код T01–T30
  model TEXT REFERENCES models(name), lang TEXT REFERENCES prompts(lang), iter INTEGER,
  temperature REAL, max_tokens INTEGER,
  status TEXT,                   -- ok / отказ / не тот язык / смесь / зацикливание
  spheres_json TEXT, explication TEXT, n_lines INTEGER, raw_response TEXT
);
CREATE TABLE scores (            -- экспертная оценка, 0–2 по каждому критерию
  code TEXT PRIMARY KEY REFERENCES runs(code),
  k1 INTEGER, k1_reason TEXT, k2 INTEGER, k3 INTEGER, k4 INTEGER, k5 INTEGER, comment TEXT
);
CREATE TABLE rubric (criterion TEXT PRIMARY KEY, name TEXT, score2 TEXT, score1 TEXT, score0 TEXT);
CREATE VIEW scored AS
  SELECT r.code, r.model, r.lang, r.iter, r.status, s.k1, s.k2, s.k3, s.k4, s.k5,
         CASE WHEN s.k2 IS NULL THEN NULL ELSE s.k1+s.k2+s.k3+s.k4+s.k5 END AS total
  FROM runs r JOIN scores s USING(code);
CREATE VIEW summary AS           -- сводка по модели и языку: среднее и выборочное SD
  SELECT model, lang, COUNT(*) AS n,
         SUM(status='отказ') AS refusals,
         ROUND(AVG(k1),2) AS k1_mean,
         ROUND(SQRT((SUM(k1*k1)-SUM(k1)*SUM(k1)*1.0/COUNT(k1))/(COUNT(k1)-1)),2) AS k1_sd,   -- выборочное SD (n−1)
         ROUND(AVG(total),2) AS total_mean,
         ROUND(SQRT((SUM(total*total)-SUM(total)*SUM(total)*1.0/COUNT(total))/(COUNT(total)-1)),2) AS total_sd
  FROM scored GROUP BY model, lang;
CREATE TABLE topics (topic INTEGER, sphere TEXT, label TEXT, share REAL, top_words TEXT);  -- 10 тем NMF и их доли
CREATE TABLE context_topics (example_id TEXT, dominant_topic INTEGER, dominant_weight REAL,
  w0 REAL, w1 REAL, w2 REAL, w3 REAL, w4 REAL, w5 REAL, w6 REAL, w7 REAL, w8 REAL, w9 REAL);  -- нормированные веса тем для 20 415 контекстов

-- Conversations and their messages. Safe to run repeatedly (IF NOT EXISTS).

CREATE TABLE IF NOT EXISTS conversations (
    id          BIGSERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS messages (
    id               BIGSERIAL PRIMARY KEY,
    -- Deleting a conversation deletes its messages too.
    conversation_id  BIGINT NOT NULL REFERENCES conversations (id) ON DELETE CASCADE,
    role             TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content          TEXT NOT NULL,
    model            TEXT,
    input_tokens     INTEGER,
    output_tokens    INTEGER,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every chat request loads one conversation's messages in order; this index makes that fast.
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages (conversation_id, id);

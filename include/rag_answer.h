#ifndef RAG_ANSWER_H
#define RAG_ANSWER_H

/*
 * rag_answer.h - answer questions straight from the table rows, without a
 *                language model (the --retrieve-only mode of llm_rag).
 *
 *   RagSession *s = rag_session_new();
 *   rag_answer(s, corpus, index, "Who is the instructor for ML?", stdout);
 *   rag_answer(s, corpus, index, "yes", stdout);     // takes the suggestion
 *   rag_session_free(s);
 *
 * Nothing here knows about courses: it works from the key columns (the
 * first two header columns), the column labels and the cell values of
 * whatever tables were loaded.
 *
 * How a question is answered. Every search term of the question (see
 * rag_index.h; typos and short forms are resolved there) is classified
 * against a candidate row as: in a key column / a word of a column label /
 * in another cell / in the title, tab or file name / filler ("course",
 * "class", "mail" ...) / missing. Then, in this order:
 *
 *   1. Follow-up   the question only names columns ("and its room?") and a
 *                  row was answered before: that row is used again.
 *   2. Lookup      rows where every term is a key term or a label word.
 *                  The row whose key column is covered best wins
 *                  ("machine learning" covers 2 of the 3 words of
 *                  "Introduction to Machine Learning"); equal winners give
 *                  a numbered choice. The same key value in several files
 *                  is one thing: the row with the most cells is used.
 *   3. Note        a footnote row containing every term is printed.
 *   4. List/count  rows where one other column contains every remaining
 *                  term ("which courses does X teach", "how many are in
 *                  slot D"): all of them are listed and counted.
 *   5. Loose       rows with a key term whose other terms appear anywhere
 *                  in the row; the best BM25 score wins.
 *   6. Otherwise   "Sorry, I couldn't find any data about your question."
 *                  and, if some rows are related (they share a key term),
 *                  an offer to list them. A term that is in no table at
 *                  all also leads here unless a key column matches fully.
 *
 * The answer is only the cells that were asked for (the columns whose
 * label matches the question, directly or through words such as "who",
 * "when", "where"), followed by one "Source:" line. Cells with e-mail
 * addresses are split into names and addresses. After an answer one more
 * column of the same row is offered ("Do you want the slot?"); the reply
 * can be yes, no, another column, or a new question.
 */

#include "rag_chunk.h"
#include "rag_index.h"

#include <stdio.h>

typedef struct RagSession RagSession;

RagSession *rag_session_new(void);
void rag_session_free(RagSession *session);

/* Forget the conversation (call after the documents were reloaded). */
void rag_session_reset(RagSession *session);

/*
 * Answer one line of input and print the reply to out. Returns 1 if an
 * answer (or list) from the tables was printed, 0 otherwise.
 */
int rag_answer(RagSession *session, const RagCorpus *corpus,
               const RagIndex *index, const char *input, FILE *out);

/*
 * Parts of the reply printed by the last rag_answer() call, valid until the
 * next call: the source ("file, tab T, row N (title)", "" if none) and the
 * closing question ("Do you want the slot?", "" if none).
 */
const char *rag_session_last_source(const RagSession *session);
const char *rag_session_last_offer(const RagSession *session);

#endif

"""VEDA demo — semantic search with no vector DB, no model, no dependencies.

Run:  python demo.py            (quick corpus demo)
      python demo.py --scale    (also ingest a few hundred KB and time it)
"""

import sys
import time

from veda import Veda

CORPUS = {
    "finance": (
        "The stock market opened sharply lower as equities plunged across "
        "every sector. Shares of major banks fell while traders rushed to "
        "sell. Analysts said the market crash wiped billions from stocks, "
        "and investors moved money into bonds to escape the falling shares. "
        "The exchange halted trading twice as equities kept sliding."
    ),
    "medicine": (
        "A doctor or physician examines patients at the hospital every "
        "morning. The physician prescribed a new treatment after reviewing "
        "the patient's symptoms, and nurses at the hospital monitored the "
        "recovery. Good clinical care depends on the physician listening "
        "carefully to each patient before choosing a therapy."
    ),
    "space": (
        "The rocket lifted off from the launch pad carrying a satellite "
        "into orbit. Engineers at mission control cheered as the spacecraft "
        "separated cleanly. The satellite will study distant galaxies and "
        "send astronomy data back to telescopes on Earth, helping "
        "scientists map the early universe."
    ),
    "cooking": (
        "Heat the oil in a heavy pan, add cumin seeds and chopped onions, "
        "and fry until golden. Stir in the tomatoes, turmeric and garam "
        "masala, then simmer the curry gently. Serve the dish hot with "
        "fresh rice and naan, garnished with coriander leaves."
    ),
}

QUERIES = [
    "stock market fall",      # lexical + semantic: finance
    "doctor treating illness",  # 'doctor' co-occurs with physician/hospital
    "satellite in orbit",     # space
    "spicy recipe with onions",  # cooking
]


def main():
    engine = Veda()
    t0 = time.time()
    for doc_id, text in CORPUS.items():
        engine.add(doc_id, text)
    print(f"Ingested {len(CORPUS)} documents in {time.time() - t0:.2f}s "
          f"(no vector DB, no model, stdlib only)\n")

    for query in QUERIES:
        print(f"Q: {query!r}")
        for hit in engine.search(query, k=2):
            print(f"   [{hit['score']:+.3f}] ({hit['doc']}) "
                  f"{hit['snippet'][:90]}...")
        print()

    if "--scale" in sys.argv:
        big = ("\n\n".join(CORPUS.values()) + "\n\n") * 60  # a few hundred KB
        engine2 = Veda()
        t0 = time.time()
        engine2.add("big", big)
        t_ingest = time.time() - t0
        t0 = time.time()
        hits = engine2.search("market crash and falling shares", k=3)
        t_query = time.time() - t0
        sigs = len(engine2.index.leaves)
        print(f"--scale: {len(big)/1024:.0f} KB ingested in {t_ingest:.1f}s, "
              f"{sigs} chunk signatures, query in {t_query*1000:.0f} ms, "
              f"top hit doc={hits[0]['doc']} score={hits[0]['score']}")


if __name__ == "__main__":
    main()

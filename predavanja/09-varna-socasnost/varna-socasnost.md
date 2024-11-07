# Varna sočasnost

- *angl.* concurrency-safe, thread safe
- vse programske strukture in metode ter s tem tudi paketi niso prilagojeni za sočasno izvajanje
- lahko prihaja do tveganih stanj
- če so programske strukture in metode deklarirane kot varne za sočasnost, ne bo prišlo do tveganih stanj

- [slovar-1.go](koda/slovar-1.go)
  - uporabimo strukturo slovar (`map`), ki ni varna za sočasnost
  - če vse sočasne gorutine samo berejo, je vse v redu
  - če ena gorutina piše in druga bere, pride do napake
  - če dve gorutini pišeta, pride do napake

- [slovar-2.go](koda/slovar-2.go)
  - lahko uporabimo bralno-pisalno ključavnico

- [slovar-3.go](koda/slovar-3.go)
  - uporabimo posebno strukturo `sync.Map`, ki je bistveno hitrejša

- paket `math/rand`
  - dostop do strukture `rand` je v jeziku go zaščiten s ključavnicami
  - strukture tipa `*rand.Rand`, ki jih ustvarimo sami, moramo pri hkratni uporabi v več gorutin varovati s ključavnicami
  - če želimo učinkovito generirati psevdonaključna števila, v vsaki gorutini naredimo svojo strukturo tipa `*rand.Rand`
  - primer neučinkovite [pi-11.go](koda/pi-11.go) in učinkovite [pi-12.go](koda/pi-12.go) kode
  - pri uporabi skupne spremenljivke ne moremo zagotavljati ponovljivosti rezultatov pri danem semenu in številu gorutin

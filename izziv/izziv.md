# Božično - novoletni izziv

## Kaj?

Razširjanje sporočil nadgradite z zagotavljanjem vrstnega reda dostave sporočil aplikaciji:

- Pripravite ogrodje za razširjanje sporočil, ki na vsakem procesu vzpostavi medpomnilnik za sporočila in vključuje metode za
  - pošiljanje sporočil,
  - sprejemanje in shranjevanje sporočil v medpomnilnik,
  - izbiranje sporočila glede na zahteve vrstnega reda dostave in
  - posredovanje izbranega sporočila aplikaciji.

- Sporočila dopolnite s potrebnimi metapodatki, ki vam bodo omogočali izbiranje sporočil za pošiljanje glede na uporabljeno shemo.

- Za komunikacijo lahko uporabite protokol UDP, protokol TCP, ali eno od rešitev za oddaljeno klicanje metod.

Podprite eno od spodnjih različic za razširjanja sporočil z zagotavljanjem vrstnega reda dostave:

- Pri **vzročnem razširjanju** nadgradite razširjanje z govoricami z mehanizmom, podobnim vektorskim uram. Sledite [psevdo algoritmu](../predavanja/14-razsirjanje-sporocil/razsirjanje-sporocil.md#algoritem-za-vzročno-razširjanje). Da zagotovimo deterministično obnašanje procesov, naj pošiljatelj iz konfiguracijske datoteke prebere oznake prejemnikov sporočila.

- Pri **popolnoma urejenem razširjanju FIFO** sledite [pristopu z enim voditeljem](../predavanja/14-razsirjanje-sporocil/razsirjanje-sporocil.md#popolnoma-urejeno-razširjanje-in-popolnoma-urejeno-razširjanje-fifo). Za povečanje odpornosti uporabite algoritem [raft](../predavanja/16-replikacija-2/replikacija-2.md#replikacija-z-voditeljem-algoritem-raft-uds9), pri tem lahko izhajate iz obstoječe [kode ali knjižnice za jezik go](../predavanja/16-replikacija-2/replikacija-2.md#raft-v-jeziku-go). Popolnoma urejeno razširjanje FIFO si lahko predstavljamo kot skupino procesov v shemi raft, kjer ni zunanjih odjemalcev - odjemalci so kar procesi v shemi raft (sledilci, kandidati, voditelj) in voditelju pošiljajo sporočila. Naloga voditelja je, da prejeta sporočila v zahtevanem vrstnem redu razširi na vse sledilce.  

Procesi naj ob zagonu preberejo konfiguracijsko datoteko v fromatu `json` (primer [izziv.json](izziv.json)). Za branje datotek lahko uporabite paket [json](https://pkg.go.dev/encoding/json). Konfiguracija vključuje:

- število procesov (`processes`),
- urnik prenašanja sporočil med procesi, ki vključuje poljubno število zapisov z naslednjimi podatki:
  - oznako pošiljatelja (`sender`),
  - seznam prejemnikov (`receivers`),
  - sporočilo (`message`),
  - čas začetka pošiljanja sporočila (`timestart`) v sekundah - ob tem času pošiljatelj sporočilo pošlje prvemu procesu v tabeli `receivers`,
  - časovni zamik (`delay`) pred pošiljanjem sporočila naslednjemu procesu v seznamu `receivers` (podan v sekundah).

Da bodo procesi kolikor toliko časovno usklajeni, bomo vse procese zaganjali na enem vozlišču. Procese oštevilčimo od 0..`processes`-1. Ob vzpostavitvi naj vsi procesi vprašajo proces $0$ za njegov začetni čas. Med izvajanjem naj potem vsi procesi sledijo urniku pošiljanja glede na začetni čas procesa $0$.

## Zakaj?

Seveda najprej zato, da porazdeljene algoritme bolje razumete in da se še bolje spoznate z jezikom go.

Potem pa tudi zato, da si zagotovite lepšo končno oceno. Končna ocena predmeta je sestavljena iz ocene domačih nalog (50 %) in iz pisnega izpita (50 %). Pravilna in lepo predstavljena rešitev za vzročno razširjanje vam h končni oceni prinese do 10 %, prepričljiva rešitev s popolnoma urejenim razširjanjem FIFO pa do 30 %.

## Kdaj?

Rešitve morate oddati najkasneje do srede, 15. 1. 2024, in jih uspešno zagovarjati pred prvim izpitnim rokom.

## Kako?

Rešitev naložite na učilnico in jo zagovarjate profesorju. Na zagovoru

- predstavite kodo in
- demonstrirate pravilnost delovanja (lahko na gruči ali na vašem prenosniku)
  - zanimivi testni primeri
  - dogodke beležite z vektorskimi urami, dnevnik vizualizirate, ...

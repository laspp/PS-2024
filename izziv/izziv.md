# Božično - novoletni izziv

## Kaj?

Razširjanje sporočil z govoricami iz tretje naloge nadgradite z zagotavljanjem vrstnega reda dostave sporočil aplikaciji.

- Pripravite ogrodje za razširjanje sporočil, ki na vsakem procesu vzpostavi medpomnilnik za sporočila in vključuje metode za

  - pošiljanje sporočil,
  - sprejemanje in shranjevanje sporočil v medpomnilnik,
  - izbiranje sporočila glede na zahteve vrstnega reda dostave in
  - posredovanje izbranega sporočila aplikaciji.

- Sporočila dopolnite s potrebnimi metapodatki, ki vam bodo omogočali izbiranje najprimernejšega sporočila.

Podprite dve različici razširjanja sporočil z zagotavljanjem vrstnega reda dostave:

- Pri **vzročnem razširjanju** uporabite koncept, podoben vektorskim uram in sledite [psevdo algoritmu](../14-razsirjanje-sporocil/razsirjanje-sporocil.md#algoritem-za-vzročno-razširjanje).
- Pri **popolnoma urejenem razširjanju FIFO** sledite [pristopu z enim voditeljem](../14-razsirjanje-sporocil/razsirjanje-sporocil.md#popolnoma-urejeno-razširjanje-in-popolnoma-urejeno-razširjanje-fifo). Za povečanje odpornosti voditelja uporabite algoritem [raft](../16-replikacija-2/replikacija-2.md#replikacija-z-voditeljem-algoritem-raft-uds9), pri tem lahko uporabite obstoječo [kodo ali knjižnico za jezik go](../16-replikacija-2/replikacija-2.md#raft-v-jeziku-go).

Procesi naj ob zagonu preberejo konfiguracijsko datoteko, ki vključuje

- število procesov,
- graf časov prenašanja sporočil v obliki matrike; vrednost 0 pomeni, da ni povezave in
- urnik odpošiljanja sporočil (proces, sporočilo, čas).

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

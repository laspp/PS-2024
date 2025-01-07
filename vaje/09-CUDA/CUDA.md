# Programiranje grafičnih procesnih enot CUDA

Na gruči Arnes je na voljo več vozlišč z računskimi karticami Nvidia [V100](https://www.nvidia.com/en-us/data-center/v100/) in [H100](https://www.nvidia.com/en-us/data-center/h100/)

Primer zagona programa `nvidia-smi` (*nvidia system management interface*) na gruči. Program izpiše podatke o računskih karticah, ki so na voljo na danem računskem vozlišču.
```Bash
$ srun --partition=gpu --reservation=fri -G1 nvidia-smi --query
```

Napišimo še lasten [program](./koda/discover-device.cu) v programskem jeziku go, ki izpiše informacije o GPE. Podpora za programiranje grafičnih procesnih enot v programskem jeziku go je omejena. Uporabili bomo paket, ki v go doda možnost zaganjanja funkcij na grafičnih procesnih enotah s pomočjo okolja CUDA in ga je v okviru diplomske naloge razvil študent FRI. Paket najdete na [repozitoriju](https://github.com/InternatBlackhole/cudago). Na repozitoriju najdete tudi nekaj primerov uporabe paketa ter krajšo dokumentacijo. 

## Namestitev CudaGo na gruči Arnes

Najprej naložimo ustrezne module:
```Bash
$ module load CUDA
$ module load Go
```
Nato nastavimo okoljski spremenljivki `CGO_CFLAGS` in `CGO_LDFLAGS`, ki ju potrebujemo za namestitev prevajalnika CudaGo.
```Bash
export CGO_CFLAGS=$(pkg-config --cflags cudart-12.6) # or other version
export CGO_LDFLAGS=$(pkg-config --libs cudart-12.6) # or other version
```

Sedaj poženemo ukaz:
```Bash
$ go install github.com/InternatBlackhole/cudago/CudaGo@latest
```
S tem namestimo prevajalnik CudaGo v mapo `~/go/bin`. Da se izognemo pisanju polne poti, ko zaganjamo prevajalnik, dodamo v okoljsko spremenljivko `$PATH` ustrezno pot:
```Bash
$ export PATH="~/go/bin/:$PATH"
```
Preverimo, če `CudaGo` deluje:
```Bash
$ CudaGo -version
```

Okoljski spremenljivki `CGO_CFLAGS` in `CGO_LDFLAGS` in `$PATH` moramo nastaviti vsakič, ko želimo uporabljati prevajalnik CudaGo. V ta namen smo vam pripravili priročno [skripto](../../predavanja//21-cuda-primeri/koda/go/cudago-init.sh), ki ustrezno inicializira okolje.

## Izpis informacij o napravi CUDA v Go

Vzamemo kodo iz [primera](./koda/deviceInfo/main.go) in jo prenesemo v poljubno mapo na gruči.
Znotraj mape ustvarimo nov modul in namestimo potrebne pakete:
```Bash
$ go mod init cudaInfo
$ go mod tidy
```
Poženemo primer:
```Bash
$ srun --partition=gpu --reservation=fri --gpus=1 go run .
```

## Zagon funkcije na GPE
Vzamemo datoteki [main.go](./koda/cudaHello/main.go) in [kernel.cu](./koda/cudaHello/kernel.cu) in ju prenesemo v poljubno mapo. Znotraj mape ustvarimo nov modul in namestimo potrebne pakete:
```Bash
$ go mod init cudaPrimer
$ go mod tidy
```

Sedaj prevedem ščepec znotraj datoteke `kernel.cu` v paket go `helloCuda`:
```Bash
$ CudaGo -package helloCuda kernel.cu
```

Poženemo primer:
```Bash
$ srun --partition=gpu --reservation=fri --gpus=1 go run .
```

## Naloga

Navodila za peto domačo nalogo najdete [tukaj](../naloga-5/naloga-5.md).

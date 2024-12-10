# Programski vmesnik CUDA

- ščepec ali jedro predstavlja kodo, ki se izvaja na GPE
- ščepec napišemo kot zaporedni program
- programski vmesnik poskrbi za prevajanje in prenos ščepca na napravo
- ščepec izvaja vsaka nit posebej na svojih podatkih

## Hierarhična organizacija niti

- mreža niti (*angl.* grid)
  - vse niti v mreži izvajajo isti ščepec
  - niti v mreži si delijo globalni pomnilnik na GPE
  - mreža je sestavljena iz blokov niti
- blok niti
  - vse niti v bloku se izvajajo na isti računski enoti
  - preko skupnega pomnilnika lahko izmenjujejo in se sinhronizirajo
- nit
  - zaporedno izvaja ščepec na svojih podatkih
  - uporablja zasebni pomnilnik
  - z nitmi v bloku si deli skupni pomnilnik
  - lahko dostopa do globalne pomnilnika in pomnilnika konstant

## Označevanje niti

- 1D, 2D ali 3D oštevilčenje
- število dimenzij izberemo glede na naravo problema
- CUDA C podpira množico spremenljivk, ki odražajo organizacijo niti
  - `threadIdx.x`, `threadIdx.y`, `threadIdx.z`
  - `blockIdx.x`, `blockIdx.y`, `blockIdx.z`
  - `blockDim.x`, `blockDim.y`, `blockDim.z`
  - `gridDim.x`, `gridDim.y`, `gridDim.z`
- niti se v snope združujejo najprej po dimenziji `x`, potem `y` in nazadnje `z`

<img src="slike/organizacija-niti.png" width="70%" />

## Ščepec ali jedro

- ščepec ali jedro je koda, ki jo zaženemo na gostitelju, izvaja pa se na napravi
- ščepec napišemo kot funkcijo, ki ji pred deklaracijo dodamo ključno besedo `__global__`
- ščepec ne vrača vrednosti
- primer ščepca

    ```C
    __global__ void pozdrav(void) {
        printf("Pozdrav z naprave, nit %d.%d!\n", blockIdx.x, threadIdx.x);
    }
    ```

- ščepec zaženemo na gostitelju, kjer med ime in argumente vrinemo trojne trikotne oklepaje
- med trojne trikotne oklepaje vpišemo organizacijo niti v mreži - število blokov in število niti v vsaki dimenziji
- za opis večdimenzionalne organizacije niti jezik CUDA C ponuja strukturo `dim3`

    ```C
    dim3 gridSize(numBlocks, 1, 1);
    dim3 blockSize(numThreads, 1, 1);
    pozdrav<<<gridSize, blockSize>>>();
    ```

- ščepec lahko kliče tudi druge funkcije na napravi, ki jih označimo s ključno besedo `__device__`
- če želimo poudariti, da se funkcija izvaja samo na gostitelju, jo označimo s `__host__`

- prvi program na GP
  - [pozdrav-gpe.cu](koda/pozdrav-gpe.cu)
  - naložimo modul: `module load CUDA`
  - kodo prevedemo s prevajalnikom za CUDA C: `nvcc -o pozdrav-gpe pozdrav-gpe.cu`
  - zaženemo program: `srun --gpus=1 --partition=gpu ./pozdrav-gpe 2 4`

## Rezervacija pomnilnika in prenašanje podatkov

- gostitelj ima dostop samo do globalnega pomnilnika naprave

### Eksplicitno prenašanje podatkov

- na gostitelju pomnilnik rezerviramo s funkcijo `malloc` in vanj vpišemo podatke
- globalni pomnilnik na napravi rezerviramo s klicem funkcije
  
  ```C
  cudaError_t cudaMalloc(void** dPtr, size_t count)
  ```

  - funkcija rezervira `count` bajtov in vrne naslov v globalnem pomnilniku naprave v kazalcu `dPtr`
- za prenašanje podatkov med globalnim pomnilnikom naprave in pomnilnikom gostitelja uporabimo funkcijo
  
  ```C
  cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, cudaMemcpyKind kind)
  ```

  - funkcija kopira `count` bajtov iz naslova `src` na naslov `dst` v smeri določeni s `kind`, ki je 
    - za prenos podatkov iz gostitelja na napravo `cudaMemcpyHostToDevice` in
    - za prenos podatkov iz naprave na gostitelja `cudaMemcpyDeviceToHost`
  - funkcija je blokirajoča - izvajanje programa se nadaljuje šele po končanem prenosu podatkov
- pomnilnik na napravi sprostimo s klicem funkcije

  ```C
  cudaError_t cudaFree(void *devPtr)
  ```

- pomnilnik na gostitelju sprostimo s klicem funkcije `free`

### Enotni pomnilnik

- novejše različice CUDA podpirajo enotni pomnilnik
- prenos podatkov izvaja CUDA po potrebi
- programer nima nadzora, večkrat manj učinkovito od eksplicitnega prenašanja
- enotni pomnilnik rezerviramo s klicem funkcije

  ```C
  cudaError_t = cudaMallocManaged(void **hdPtr, count);
  ```

- enotni pomnilnik sprostimo s klicem funkcije

  ```C
  cudaError_t cudaFree(void *hdPtr)
  ```

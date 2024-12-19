#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>

int numLinhas;      //variável para armazenar o número de linhas de um arquivo lido
int numLinhasTest;  //armazenar o numero de linhas de teste
int numLinhasTrain; //armazenar o numero de linhas de treino

int w;  // janela de observacoes
int h;  // horizonte de previsão
int k;  // número de vizinhos

//função para processar os arquivos (ler e armazenar os valores em um array) e contar o numero de linhas
int lerArquivo(const char *filename, double *array) {
    FILE *arquivo = fopen(filename, "r");
    if (arquivo == NULL) {
        perror("Erro ao abrir o arquivo");      //verificação de erro
        return -1;
    }

    numLinhas = 0;
    while (fscanf(arquivo, "%lf", &array[numLinhas]) == 1) {  //lê valores flutuantes enquanto possível
        numLinhas++;
    }

    fclose(arquivo);
    return 0;
}

//função para montar as matrizes Xtrain e Xtest (de acordo como foi especificado para o ep1)
float** montaX(double *vetor, double **matriz) {
    int indice = 0;
    for (int i = 0; i < (numLinhas - w - h + 1); i++) {
        for (int j = 0; j < w; j++) {
            matriz[i][j] = vetor[indice];
            indice++;
        }
        indice -= w - 1 ;
    }

    return 0;
}

//função para montar a matriz Ytrain (de acordo como foi especificado para o ep1)
double* montaY(double *vetor) {
    double *vetory = malloc((numLinhas - w - h + 1) * sizeof(double));
    for (int i = 0; i < (numLinhas - w - h + 1); i++) {
        vetory[i] = vetor[i + w + h - 1];
    }

    return vetory;
} 

//função para alocar memória para as matrizes
double ** alocaMatriz() {
    double **matriz = malloc((numLinhas - w - h + 1) * sizeof(double*));
    for (int i = 0; i < (numLinhas - w - h + 1); i++) {
        matriz[i] = malloc(w * sizeof(double));
    }
    return matriz;
}

//função para calculo da distância euclidiana (entre teste e treino)
double* calculaDistancia(double **xtrain,  double **xtest, int linhaAtual) {
    double *distancias = malloc((numLinhasTrain - w - h + 1) * sizeof(double));
    double d = 0.0;
    for(int l = 0; l < (numLinhasTrain - w - h + 1); l++) {
        for(int c = 0; c < w; c++) {
            d += pow(xtrain[l][c] - xtest[linhaAtual][c], 2);
        }
        distancias[l] = d;
        d = 0.0;
    }

    return distancias;
}

//função para encontrar os indices das k menores distâncias para utilização no knn
void k_menores_indices(double *distancias, int *indices) {
    double *menores = malloc(k * sizeof(double));
    int *indicesMenores = malloc(k * sizeof(int));

    for (int i = 0; i < k; i++) {
        menores[i] = DBL_MAX;       //inicialização dos vetores
        indicesMenores[i] = -1;
    }

    for (int i = 0; i < (numLinhasTrain - w - h + 1); i++) {        //percorre o vetor de distancias
        if (distancias[i] < menores[k - 1]) {
            int j = k - 1;
            while (j > 0 && distancias[i] < menores[j - 1]) {
                menores[j] = menores[j - 1];        
                indicesMenores[j] = indicesMenores[j - 1];          //ordenaçao dos k menores valores
                j--;
            }
            menores[j] = distancias[i];                 //preenchimentos dos vetores com os valores corretos
            indicesMenores[j] = i;
        }
    }

    for (int i = 0; i < k; i++) {
        indices[i] = indicesMenores[i];     
    }

    free(menores);
    free(indicesMenores);
}

//função knn
double* knn (double **xtrain, double *ytrain, double **xtest) {
    double *vetorytest = malloc((numLinhasTest - w - h + 1) * sizeof(double));

    for(int x = 0; x < (numLinhasTest - w - h + 1); x++) {       //loop para percorrer xtest
        double *vetorDistancias = calculaDistancia(xtrain, xtest, x);       //cálculo das distancias da linha de teste atual
        int *indices = malloc(k * sizeof(int));     
        k_menores_indices(vetorDistancias, indices);            //identificação dos k menores indices para a linha atual

        //cálculo da rotação correta
        if(k == 1) {
            vetorytest[x] = ytrain[indices[0]];
        }
        
        if(k > 1) {
            double media = 0.0;
            for(int t = 0; t < k; t++) {
                media += ytrain[indices[t]];
            }
            media /= k;
            vetorytest[x] = media;
        }
        free(vetorDistancias);
        free(indices);
    }

    return vetorytest;
}

//função para escrever o vetor ytest em um arquivo
void salvarArrayEmArquivo(const double array[], const char *nomeArquivo) {
    FILE *arquivo = fopen(nomeArquivo, "w");

    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo.\n");       //verificação de erro
        return;
    }

    for (int i = 0; i < numLinhas-w-h+1; i++) {
        fprintf(arquivo, "%.6lf", array[i]);
        if (i != (numLinhas-w-h)) {
            fprintf(arquivo, "\n");                 //escrita no arquivo até o último valor
        }
    }

    fclose(arquivo);
}

//main
int main (int argc, char *argv[]) {
    if (argc < 6) {
        printf("Execute fornecendo os parametros (nessa ordem): %s <k> <w> <h> <arquivo_Xtrain> <arquivo_Xtest>\n", argv[0]);
        return 1;
    }

    // Lendo os parâmetros da linha de comando
    k = atoi(argv[1]);
    w = atoi(argv[2]);
    h = atoi(argv[3]);

    const char *arquivoXtrain = argv[4];
    const char *arquivoXtest = argv[5];

    clock_t start, end;     //registros para a contagem de tempo
    double elapsed_time;    //tempo final

    double *array = malloc(10000000 * sizeof(double));  //array de entrada dos arquivos lidos

    lerArquivo(arquivoXtrain, array);        //leitura de Xtrain
    numLinhasTrain = numLinhas;
    double **matrizxtrain = alocaMatriz();  //aloca memória para a construção da matriz Xtrain
    montaX(array, matrizxtrain);        //montagem da matriz Xtrain

    double *vetorytrain = montaY(array);        //montagem do vetor ytrain

    free(array);

    double *array2 = malloc(10000000 * sizeof(double)); //array de entrada dos arquivos lidos

    lerArquivo(arquivoXtest, array2);    //leitura de Xtest
    numLinhasTest = numLinhas;
    double **matrizxtest = alocaMatriz();   //aloca memória para a construção da matriz Xtest
    montaX(array2, matrizxtest);        //montagem da matriz Xtest

    free(array2);

    start = clock();        //inicio da contagem tempo

    double *vetorytest = knn(matrizxtrain, vetorytrain, matrizxtest);       //chamada da funçao knn

    end = clock();          //fim da contagem de tempo
    
    free(matrizxtrain);
    free(vetorytrain);
    free(matrizxtest);

    salvarArrayEmArquivo(vetorytest, "Ytest.txt");      //salvar ytest em um arquivo

    free(vetorytest);

    elapsed_time = ((double)(end - start)) / CLOCKS_PER_SEC;        //cálculo do tempo de execução de knn
    printf("Tempo de execucao: %.3lf segundos\n", elapsed_time);
    
    return 0;
}

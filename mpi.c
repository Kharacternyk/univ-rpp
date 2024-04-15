#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gmp.h>

static void master(int argc, char **argv) {
  assert(argc == 3);

  unsigned long input_string_length = strlen(argv[1]);
  MPI_Bcast(&input_string_length, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
  MPI_Bcast(argv[1], input_string_length, MPI_BYTE, 0, MPI_COMM_WORLD);

  long seed = strtol(argv[2], NULL, 0);
  srand48(seed);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  for (int i = 1; i < world_size; ++i) {
    long initial_x = lrand48();
    MPI_Send(&initial_x, 1, MPI_LONG, i, 0, MPI_COMM_WORLD);
  }
}

static void polynomial(mpz_t x, const mpz_t modulus) {
  mpz_mul(x, x, x);
  mpz_add_ui(x, x, 1);
  mpz_cdiv_r(x, x, modulus);
}

static void worker() {
  unsigned long input_string_length;
  MPI_Bcast(&input_string_length, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

  char *input_string = malloc(input_string_length + 1);
  MPI_Bcast(input_string, input_string_length, MPI_CHAR, 0, MPI_COMM_WORLD);
  input_string[input_string_length] = 0;

  mpz_t input;
  mpz_init_set_str(input, input_string, 10);

  long initial_x;
  MPI_Recv(&initial_x, 1, MPI_LONG, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  printf("Searching for a prime factor of %s with initial x = %ld\n",
         input_string, initial_x);

  mpz_t x;
  mpz_init_set_si(x, initial_x);

  mpz_t y;
  mpz_init_set(y, x);

  mpz_t divisor;
  mpz_init_set_ui(divisor, 1);

  while (!mpz_cmp_ui(divisor, 1)) {
    polynomial(x, input);
    polynomial(y, input);
    polynomial(y, input);

    mpz_sub(divisor, x, y);
    mpz_abs(divisor, divisor);
    mpz_gcd(divisor, divisor, input);
  }

  printf("Found a prime factor ");
  mpz_out_str(NULL, 10, divisor);
  printf(" with initial x = %ld\n", initial_x);

  MPI_Abort(MPI_COMM_WORLD, 0);
}

int main(int argc, char **argv) {
  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    master(argc, argv);
  } else {
    worker();
  }

  MPI_Finalize();
}

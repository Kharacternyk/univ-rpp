#include <assert.h>
#include <omp.h>
#include <stdbool.h>
#include <stdio.h>

#include <gmp.h>

static void polynomial(mpz_t x, const mpz_t modulus) {
  mpz_mul(x, x, x);
  mpz_add_ui(x, x, 1);
  mpz_cdiv_r(x, x, modulus);
}

int main(int argc, char **argv) {
  assert(argc == 4);

  int thread_count = strtol(argv[1], NULL, 10);
  omp_set_num_threads(thread_count);

  mpz_t input;
  mpz_init_set_str(input, argv[2], 10);

  long seed = strtol(argv[3], NULL, 10);
  srand48(seed);

  long *initial_xs = calloc(thread_count, sizeof(long));
  for (int i = 0; i < thread_count; ++i) {
    initial_xs[i] = lrand48();
  }

  bool done = false;

#pragma omp parallel
  {
    long initial_x = initial_xs[omp_get_thread_num()];
    printf("Searching for a prime factor of %s with initial x = %ld\n", argv[2],
           initial_x);

    mpz_t x;
    mpz_init_set_si(x, initial_x);

    mpz_t y;
    mpz_init_set(y, x);

    mpz_t divisor;
    mpz_init_set_ui(divisor, 1);

    while (!done && !mpz_cmp_ui(divisor, 1)) {
      polynomial(x, input);
      polynomial(y, input);
      polynomial(y, input);

      mpz_sub(divisor, x, y);
      mpz_abs(divisor, divisor);
      mpz_gcd(divisor, divisor, input);
    }

    if (!done) {
      done = true;

      printf("Found a prime factor ");
      mpz_out_str(NULL, 10, divisor);
      printf(" with initial x = %ld\n", initial_x);
    }
  }
}

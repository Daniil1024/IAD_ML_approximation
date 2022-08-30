static void print_results_header(FILE *fp);
void print_optical_property_result(FILE *fp,
								   struct measure_type m,
								   struct invert_type r,
								   double LR,
								   double LT,
								   double mu_a,
								   double mu_sp, int mc_iter, int line);
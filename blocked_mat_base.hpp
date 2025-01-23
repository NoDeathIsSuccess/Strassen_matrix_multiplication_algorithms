#include <vector>
#include <random>
#include <iostream>
#include <string>

constexpr int rdrange = 100;
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dis(-rdrange, rdrange);

//void read_mat(double*** A, double*** B, size_t* m1, size_t* m2, size_t* m3, const std::string& finame) {
//	uint64_t n1, n2, n3;
//	FILE* fi;
//
//	fi = fopen(finame.c_str(), "rb");
//	fread(&n1, 1, 8, fi);
//	fread(&n2, 1, 8, fi);
//	fread(&n3, 1, 8, fi);
//
//	double* a = (double*)malloc(n1 * n2 * sizeof(double));
//	double* b = (double*)malloc(n2 * n3 * sizeof(double));
//
//	fread(a, 1, n1 * n2 * sizeof(double), fi);
//	fread(b, 1, n2 * n3 * sizeof(double), fi);
//	*A = (double**)malloc(sizeof(double*) * n1);
//	for (size_t i = 0;i < n1;++i) {
//		(*A)[i] = a + i * n2;
//	}
//	*B = (double**)malloc(sizeof(double*) * n2);
//	for (size_t i = 0;i < n2;++i) {
//		(*B)[i] = b + i * n3;
//	}
//	*m1 = n1;
//	*m2 = n2;
//	*m3 = n3;
//	fclose(fi);
//}

double** create_mat(size_t n, size_t m) {
	double* p = (double*)malloc(sizeof(double) * n * m);
	double** ans = (double**)malloc(sizeof(double*) * n);
	for (size_t i = 0;i < n;++i) {
		ans[i] = p + i * m;
		for (size_t j = 0;j < m;++j) {
			p[i * m + j] = dis(gen);
		}
	}
	return ans;
}

double** create_null_mat(size_t n, size_t m) {
	double* p = (double*)malloc(sizeof(double) * n * m);
	double** ans = (double**)malloc(sizeof(double*) * n);
	for (size_t i = 0;i < n;++i) {
		ans[i] = p + i * m;
	}
	return ans;
}

void print_mat(const double* const* mat, size_t n, size_t m) {
	for (int i = 0;i < n;++i) {
		for (int j = 0;j < m;++j) {
			std::cout << mat[i][j] << ' ';
		}
		std::cout << '\n';
	}
	std::cout << std::endl;
}

void save_mat(const double* const* mat, size_t n, size_t m, const std::string& finame) {
	FILE* fi;
	fi = fopen(finame.c_str(), "wb");
	for (int i = 0;i < n;++i) {
		fwrite(mat[i], 1, sizeof(double) * m, fi);
	}
	fclose(fi);
}

void free_mat(double** A) {
	free(A[0]);
	free(A);
}
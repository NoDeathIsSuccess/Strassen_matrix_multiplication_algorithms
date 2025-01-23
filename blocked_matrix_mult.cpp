#define _CRT_SECURE_NO_WARNINGS
#include "blocked_mat_base.hpp"
#include <iostream>
#include <set>
#include <chrono>
#include <omp.h>

auto start_time = std::chrono::high_resolution_clock::now();
auto end_time = std::chrono::high_resolution_clock::now();
auto duration_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

constexpr size_t goalsize = 512;

double** mat_transpose(double** mat, size_t xbeg, size_t xend, size_t ybeg, size_t yend) {
	size_t xlen = xend - xbeg;
	size_t ylen = yend - ybeg;
	/*double* pans = (double*)malloc(sizeof(double) * xlen * ylen);
	double** ans = (double**)malloc(sizeof(double*) * ylen);
	for (int i = 0;i < ylen;++i) {
		ans[i] = pans + i * xlen;
	}*/
	double** ans = create_null_mat(xlen, ylen);
#pragma omp parallel for collapse(2)
	for (int i = 0; i < xlen; ++i) {
		for (int j = 0; j < ylen; ++j) {
			ans[j][i] = mat[i + xbeg][j + ybeg];
		}
	}
	return ans;
}

double** base_mat_mult(double** A, double** B,
	size_t Axbeg, size_t Axend, size_t Aybeg, size_t Ayend,
	size_t Bxbeg, size_t Bxend, size_t Bybeg, size_t Byend) {
	size_t Ax = Axend - Axbeg;
	size_t Ay = Ayend - Aybeg;
	size_t Bx = Bxend - Bxbeg;
	size_t By = Byend - Bybeg;
	double** Btrans = mat_transpose(B, Bxbeg, Bxend, Bybeg, Byend);
	/*double* pans = (double*)malloc(sizeof(double) * Ax * By);
	memset(pans, 0, sizeof(double) * Ax * By);
	double** ans = (double**)malloc(sizeof(double*) * Ax);
	for (int i = 0;i < Ax;++i) {
		ans[i] = pans + i * By;
	}*/
	double** ans = create_null_mat(Ax, By);
#pragma omp parallel for collapse(2)
	for (int i = 0;i < Ax;++i) {
		for (int j = 0;j < By;++j) {
			double tempans = 0;
#pragma omp simd reduction(+: tempans)
			for (int k = 0;k < Ay;++k) {
				tempans += A[i + Axbeg][k + Axbeg] * Btrans[j][k];
			}
			ans[i][j] = tempans;
		}
	}
	free(Btrans);
	return ans;
}

double** mat_plus(double** A, size_t Axbeg, size_t Axend, size_t Aybeg, size_t Ayend, double** B, size_t Bxbeg, size_t Bxend, size_t Bybeg, size_t Byend, int sgn) {
	size_t Ax = Axend - Axbeg;
	size_t Ay = Ayend - Aybeg;
	if (Ax != Bxend - Bxbeg || Ay != Byend - Bybeg) { 
		exit(1); 
	}
	if (sgn == 1) {
		double** ans = create_null_mat(Ax, Ay);
#pragma omp parallel for collapse(2)
		for (int i = 0;i < Ax;++i) {
			for (int j = 0;j < Ay;++j) {
				ans[i][j] = A[Axbeg + i][Aybeg + j] + B[Bxbeg + i][Bybeg + j];
			}
		}
		return ans;
	}
	else if (sgn == -1) {
		double** ans = create_null_mat(Ax, Ay);
#pragma omp parallel for collapse(2)
		for (int i = 0;i < Ax;++i) {
			for (int j = 0;j < Ay;++j) {
				ans[i][j] = A[Axbeg + i][Aybeg + j] - B[Bxbeg + i][Bybeg + j];
			}
		}
		return ans;
	}
	else {
		return 0;
	}
}

void mat_plus_equ(double** A, size_t Axbeg, size_t Axend, size_t Aybeg, size_t Ayend, double** B, size_t Bxbeg, size_t Bxend, size_t Bybeg, size_t Byend, int sgn) {
	size_t Ax = Axend - Axbeg;
	size_t Ay = Ayend - Aybeg;
	if (sgn == 1) {
#pragma omp parallel for collapse(2)
		for (int i = 0;i < Ax;++i) {
			for (int j = 0;j < Ay;++j) {
				A[Axbeg + i][Aybeg + j] += B[Bxbeg + i][Bybeg + j];
			}
		}
	}
	else if (sgn == -1) {
#pragma omp parallel for collapse(2)
		for (int i = 0;i < Ax;++i) {
			for (int j = 0;j < Ay;++j) {
				A[Axbeg + i][Aybeg + j] -= B[Bxbeg + i][Bybeg + j];
			}
		}
	}
}

double** split_mat_mult(double** A, double** B, 
	size_t Axbeg, size_t Axend, size_t Aybeg, size_t Ayend,
	size_t Bxbeg, size_t Bxend, size_t Bybeg, size_t Byend) {
	size_t Ax = Axend - Axbeg;
	size_t Ay = Ayend - Aybeg;
	size_t Bx = Bxend - Bxbeg;
	size_t By = Byend - Bybeg;
	if (Ay != Bx) {
		exit(1);
	}
	if (Ax <= goalsize || Ay <= goalsize || Bx <= goalsize || By <= goalsize) {
		return base_mat_mult(A, B, Axbeg, Axend, Aybeg, Ayend, Bxbeg, Bxend, Bybeg, Byend);
	}

	double** ans = create_null_mat(Ax, By);
	memset(ans[0], 0, sizeof(double) * Ax * By);

	size_t Axhalf = (Axbeg + Axend) / 2;
	size_t Ayhalf = (Aybeg + Ayend) / 2;
	size_t Bxhalf = (Bxbeg + Bxend) / 2;
	size_t Byhalf = (Bybeg + Byend) / 2;

	double** S1 = mat_plus(B, Bxbeg, Bxhalf, Byhalf, Byend, B, Bxhalf, Bxend, Byhalf, Byend, -1);//B12-B22
	double** S2 = mat_plus(A, Axbeg, Axhalf, Aybeg, Ayhalf, A, Axbeg, Axhalf, Ayhalf, Ayend, 1);//A11+A12
	double** S3 = mat_plus(A, Axhalf, Axend, Aybeg, Ayhalf, A, Axhalf, Axend, Ayhalf, Ayend, 1);//A21+A22
	double** S4 = mat_plus(B, Bxhalf, Bxend, Bybeg, Byhalf, B, Bxbeg, Bxhalf, Bybeg, Byhalf, -1);//B21-B11
	double** S5 = mat_plus(A, Axbeg, Axhalf, Aybeg, Ayhalf, A, Axhalf, Axend, Ayhalf, Ayend, 1);//A11+A22
	double** S6 = mat_plus(B, Bxbeg, Bxhalf, Bybeg, Byhalf, B, Bxhalf, Bxend, Byhalf, Byend, 1);//B11+B22
	double** S7 = mat_plus(A, Axbeg, Axhalf, Ayhalf, Ayend, A, Axhalf, Axend, Ayhalf, Ayend, -1);//A12-A22
	double** S8 = mat_plus(B, Bxhalf, Bxend, Bybeg, Byhalf, B, Bxhalf, Bxend, Byhalf, Byend, 1);//B21+B22
	double** S9 = mat_plus(A, Axbeg, Axhalf, Aybeg, Ayhalf, A, Axhalf, Axend, Aybeg, Ayhalf, -1);//A11-A21
	double** S10 = mat_plus(B, Bxbeg, Bxhalf, Bybeg, Byhalf, B, Bxbeg, Bxhalf, Byhalf, Byend, 1);//B11+B12

	//(Ax/2)*(By/2)
	double** P1 = split_mat_mult(A, S1,
		Axbeg, Axhalf, Aybeg, Ayhalf,
		0, Bx / 2, 0, By / 2);//A11S1
	double** P2 = split_mat_mult(S2, B, 
		0, Ax / 2, 0, Ay / 2,
		Bxhalf, Bxend, Byhalf, Byend);//S2B22
	double** P3 = split_mat_mult(S3, B, 
		0, Ax / 2, 0, Ay / 2,
		Bxbeg, Bxhalf, Bybeg, Byhalf);//S3B11
	double** P4 = split_mat_mult(A, S4, 
		Axhalf, Axend, Ayhalf, Ayend,
		0, Bx / 2, 0, By / 2);//A22S4
	double** P5 = split_mat_mult(S5, S6, 
		0, Ax / 2, 0, Ay / 2,
		0, Bx / 2, 0, By / 2);//S5S6
	double** P6 = split_mat_mult(S7, S8, 
		0, Ax / 2, 0, Ay / 2,
		0, Bx / 2, 0, By / 2);//S7S8
	double** P7 = split_mat_mult(S9, S10, 
		0, Ax / 2, 0, Ay / 2,
		0, Bx / 2, 0, By / 2);//S9S10

	//C11=P5+P4-P2+P6
	mat_plus_equ(ans, 0, Ax / 2, 0, By / 2, P5, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, 0, Ax / 2, 0, By / 2, P4, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, 0, Ax / 2, 0, By / 2, P2, 0, Ax / 2, 0, By / 2, -1);
	mat_plus_equ(ans, 0, Ax / 2, 0, By / 2, P6, 0, Ax / 2, 0, By / 2, 1);
	//C12=P1+P2
	mat_plus_equ(ans, 0, Ax / 2, By / 2, By, P1, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, 0, Ax / 2, By / 2, By, P2, 0, Ax / 2, 0, By / 2, 1);
	//C21=P3+P4
	mat_plus_equ(ans, Ax / 2, Ax, 0, By / 2, P3, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, Ax / 2, Ax, 0, By / 2, P4, 0, Ax / 2, 0, By / 2, 1);
	//C22=P5+P1-P3-P7
	mat_plus_equ(ans, Ax / 2, Ax, By / 2, By, P5, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, Ax / 2, Ax, By / 2, By, P1, 0, Ax / 2, 0, By / 2, 1);
	mat_plus_equ(ans, Ax / 2, Ax, By / 2, By, P3, 0, Ax / 2, 0, By / 2, -1);
	mat_plus_equ(ans, Ax / 2, Ax, By / 2, By, P7, 0, Ax / 2, 0, By / 2, -1);

	free_mat(S1);
	free_mat(S2);
	free_mat(S3);
	free_mat(S4);
	free_mat(S5);
	free_mat(S6);
	free_mat(S7);
	free_mat(S8);
	free_mat(S9);
	free_mat(S10);
	free_mat(P1);
	free_mat(P2);
	free_mat(P3);
	free_mat(P4);
	free_mat(P5);
	free_mat(P6);
	free_mat(P7);

	return ans;
}

double** blocked_matrix_mult(double** A, size_t nA, size_t mA, double** B, size_t nB, size_t mB) {
	if (nA == 0 || mA == 0 || mA != nB || mB == 0) { return 0; }
	start_time = std::chrono::high_resolution_clock::now();
	double** ans;
	ans = split_mat_mult(A, B, 
		0, nA, 0, mA,
		0, nB, 0, mB);
	end_time = std::chrono::high_resolution_clock::now();

	duration_time = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
	std::cout << "n1 = " << nA << " , n2 = " << mA << " , n3 = " << mB << '\n';
	std::cout << "cpu time: " << duration_time.count() << " microseconds" << '\n';
	std::cout << "          " << duration_time.count() / 1e6 << " seconds" << '\n';

	return ans;
}

//void do_mat_mult(const std::string& fimark) {
//	double** matA, ** matB;
//	size_t n1, n2, n3;
//	read_mat(&matA, &matB, &n1, &n2, &n3, "D:\\vs2022_cppprj\\hpcgame\\g02\\matrix_mult\\conf_" + fimark + ".data");
//	double** matC = blocked_matrix_mult(matA, n1, n2, matB, n2, n3);
//	auto duration_omp = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
//	std::cout << "calc time:\t" << duration_omp.count() << " microseconds" << '\n';
//	std::cout << "           \t" << duration_omp.count() / 1e6 << " seconds" << std::endl;
//	save_mat(matC, n1, n3, "out_" + fimark + ".data");
//	free_mat(matA);
//	free_mat(matB);
//	free_mat(matC);
//}


int main()
{
	omp_set_num_threads(8);
	//for (int i = 0;i < 4;++i) {
	//	std::cout << "file " << i + 1 << ":" << std::endl;
	//	do_mat_mult(std::to_string(i + 1));
	//}

	size_t n1, n2, n3;
	n1 = 1 << 13;
	n2 = 1 << 14;
	n3 = 1 << 13;
	double** A = create_mat(n1, n2);
	double** B = create_mat(n2, n3);
	//print_mat(A, n1, n2);
	//print_mat(B, n2, n3);
	double** C = blocked_matrix_mult(A, n1, n2, B, n2, n3);
	//print_mat(C, n1, n3);
	//save_mat(C, n1, n3, "matC.dat");
	free_mat(A);
	free_mat(B);
	free_mat(C);
}

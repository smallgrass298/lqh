/*=================================================================================
 * Difference Scheme:
 *			1st-order Upwind Scheme
 *			2nd-order UNO Scheme
 *			5th-order WENO Scheme
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>
#include <algorithm>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Array.hpp"
#include "Global.hpp"

using namespace std;
using namespace ARRAY;
/*---------------------------------------------------------------------------------
	Name:		Limiter

	Input(1):	The value needed to be corrected

	Function:	Use the limiter to correct the value

	Return:		the value after limiter correction
 *---------------------------------------------------------------------------------*/
double Minmod(double R)
{
	double Minmod = 0.0;

	if (R > 0)
		Minmod = min(R, 1.0);
	else
		Minmod = 0.0;

	return Minmod;
}

double Van_Leer(double R)
{
	double Van_Leer = 0.0;

	Van_Leer = (R + abs(R)) / (1.0 + abs(R));

	return Van_Leer;
}

double Superbee(double R)
{
	double Superbee = 0.0;

	Superbee = max({0.0, min(2 * R, 1.0), min(R, 2.0)});

	return Superbee;
}

double Van_Albada(double R)
{
	double Van_Albada = 0.0;

	Van_Albada = (pow(R, 2) + R) / (1.0 + pow(R, 2));

	return Van_Albada;
}

double Double_Minmod(double R)
{
	double Double_Minmod = 0.0;

	if (R > 0)
		Double_Minmod = min({2 * R, 1.0, (1 + R) / 2});
	else
		Double_Minmod = 0.0;

	return Double_Minmod;
}

/*---------------------------------------------------------------------------------
	Name:		Diff_Initial

	Input(7):	The direction; The mesh nodes and boundary nodes;
				The parameter of the flow field; The limiter

	Function:	Initialize the difference function

	Return:		value (no practical meaning, just for initialization)
 *---------------------------------------------------------------------------------*/

void Diff_Initial(int direction, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, Array<double, 4> &FLR, Array<double, 3> &fi, int bc, int limit)
{
}

/*---------------------------------------------------------------------------------
	Name:		MUSCL_1

	Input(7):	The direction; The mesh nodes and boundary nodes;
				The parameter of the flow field; The limiter

	Function:	Use the 1st-order MUSCL Scheme to interpolating

	Return:		the left and right value of the parameter of each node
 *---------------------------------------------------------------------------------*/

void MUSCL_1(int direction, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, Array<double, 4> &FLR, Array<double, 3> &fi, int bc, int limit)
{
	int ni = xnode.GetSize() - 2 * bc;
	int nj = ynode.GetSize() - 2 * bc;
	int nk = znode.GetSize() - 2 * bc;

	for (int i = bc - 1; i < ni + bc; i++)
		for (int j = bc - 1; j < nj + bc; j++)
			for (int k = bc - 1; k < nk + bc; k++)
			{
				if (direction == 1)
				{
					FLR(i, j, k, 0) = fi(i, j, k);
					FLR(i, j, k, 1) = fi(i + 1, j, k);
				}
				else if (direction == 2)
				{
					FLR(i, j, k, 0) = fi(i, j, k);
					FLR(i, j, k, 1) = fi(i, j + 1, k);
				}
				else if (direction == 3)
				{
					FLR(i, j, k, 0) = fi(i, j, k);
					FLR(i, j, k, 1) = fi(i, j, k + 1);
				}
			}
}

/*---------------------------------------------------------------------------------
	Name:		MUSCL_2

	Input(7):	The direction; The mesh nodes and boundary nodes;
				The parameter of the flow field; The limiter

	Function:	Use the 2nd-order MUSCL Scheme to interpolating

	Return:		the left and right value of the parameter of each node
 *---------------------------------------------------------------------------------*/

void MUSCL_2(int direction, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, Array<double, 4> &FLR, Array<double, 3> &fi, int bc, int limit)
{
	int ni = xnode.GetSize() - 2 * bc;
	int nj = ynode.GetSize() - 2 * bc;
	int nk = znode.GetSize() - 2 * bc;

	double kk = 1.0 / 3.0;
	double A0 = 0.0, A1 = 0.0, A2 = 0.0, L = 0.0, R = 0.0;
	double L1 = 0.0, L2 = 0.0, R1 = 0.0, R2 = 0.0;

	// //#pragma omp parallel for num_threads(num_thread) collapse(3) schedule(static)
	for (int i = bc - 1; i < ni + bc; i++)
		for (int j = bc - 1; j < nj + bc; j++)
			for (int k = bc - 1; k < nk + bc; k++)
			{
				// double A0 = 0.0, A1 = 0.0, A2 = 0.0, L = 0.0, R = 0.0;
				// double L1 = 0.0, L2 = 0.0, R1 = 0.0, R2 = 0.0;
				if (direction == 1)
				{
					A0 = fi(i, j, k) - fi(i - 1, j, k);
					A1 = fi(i + 1, j, k) - fi(i, j, k);
					A2 = fi(i + 2, j, k) - fi(i + 1, j, k);
				}
				else if (direction == 2)
				{
					A0 = fi(i, j, k) - fi(i, j - 1, k);
					A1 = fi(i, j + 1, k) - fi(i, j, k);
					A2 = fi(i, j + 2, k) - fi(i, j + 1, k);
				}
				else if (direction == 3)
				{
					A0 = fi(i, j, k) - fi(i, j, k - 1);
					A1 = fi(i, j, k + 1) - fi(i, j, k);
					A2 = fi(i, j, k + 2) - fi(i, j, k + 1);
				}

				L = A1 / (A0 + 1e-12);
				R = A1 / (A2 + 1e-12);

				switch (limit)
				{
				case 0:
					L1 = Minmod(L);
					L2 = Minmod(1.0 / L);
					R1 = Minmod(R);
					R2 = Minmod(1.0 / R);
					break;
				case 1:
					L1 = Van_Leer(L);
					L2 = Van_Leer(1.0 / L);
					R1 = Van_Leer(R);
					R2 = Van_Leer(1.0 / R);
					break;
				case 2:
					L1 = Van_Albada(L);
					L2 = Van_Albada(1.0 / L);
					R1 = Van_Albada(R);
					R2 = Van_Albada(1.0 / R);
					break;
				case 3:
					L1 = Superbee(L);
					L2 = Superbee(1.0 / L);
					R1 = Superbee(R);
					R2 = Superbee(1.0 / R);
					break;
				case 4:
					L1 = Double_Minmod(L);
					L2 = Double_Minmod(1.0 / L);
					R1 = Double_Minmod(R);
					R2 = Double_Minmod(1.0 / R);
					break;
				}

				if (direction == 1)
				{
					FLR(i, j, k, 0) = fi(i, j, k) + 0.25 * ((1.0 - kk) * L1 + (1 + kk) * L2 * L) * A0;
					FLR(i, j, k, 1) = fi(i + 1, j, k) - 0.25 * ((1.0 - kk) * R1 + (1 + kk) * R2 * R) * A2;
				}
				else if (direction == 2)
				{
					FLR(i, j, k, 0) = fi(i, j, k) + 0.25 * ((1.0 - kk) * L1 + (1 + kk) * L2 * L) * A0;
					FLR(i, j, k, 1) = fi(i, j + 1, k) - 0.25 * ((1.0 - kk) * R1 + (1 + kk) * R2 * R) * A2;
				}
				else if (direction == 3)
				{
					FLR(i, j, k, 0) = fi(i, j, k) + 0.25 * ((1.0 - kk) * L1 + (1 + kk) * L2 * L) * A0;
					FLR(i, j, k, 1) = fi(i, j, k + 1) - 0.25 * ((1.0 - kk) * R1 + (1 + kk) * R2 * R) * A2;
				}
			}
}

/*---------------------------------------------------------------------------------
	Name:		WENO ( Weighted Essentially Non-Oscillatory )

	Input(4):	The time step; The velocity;
				The mesh nodes; The flow field

	Function:	Use the 5th-order WENO Scheme to solve the equation

	Return:		dfi/dx
 *---------------------------------------------------------------------------------*/

// Array<double> WENO(int flag, Array<double> U, Array<double> V, Array<double> xnode, Array<double> ynode, Array<double> fi) {
//	int ni = xnode.GetSize();
//	int nj = ynode.GetSize();
//
//	Array<double> f;
//	f.Initial(ni, nj);
//
//	double e = pow(0.1, 6);        /* Epsilon */
//	double v1 = 0, v2 = 0, v3 = 0, v4 = 0, v5 = 0;  /* dfi/dx */
//	double S1 = 0, S2 = 0, S3 = 0;      /* Smooth function */
//	double a1 = 0, a2 = 0, a3 = 0;
//	double w1 = 0, w2 = 0, w3 = 0;      /* Weight */
//
//	if (flag == 1) {
//		for (int i = 3; i < ni - 3; i++)
//			for (int j = 3; j < nj - 3; j++)
//			{
//				double dx = xnode(i) - xnode(i - 1);
//				if (U(i, j) > 0.0) {
//					v1 = (fi(i - 2, j) - fi(i - 3, j)) / dx;
//					v2 = (fi(i - 1, j) - fi(i - 2, j)) / dx;
//					v3 = (fi(i, j) - fi(i - 1, j)) / dx;
//					v4 = (fi(i + 1, j) - fi(i, j)) / dx;
//					v5 = (fi(i + 2, j) - fi(i + 1, j)) / dx;
//				}
//				else {
//					v1 = (fi(i + 3, j) - fi(i + 2, j)) / dx;
//					v2 = (fi(i + 2, j) - fi(i + 1, j)) / dx;
//					v3 = (fi(i + 1, j) - fi(i, j)) / dx;
//					v4 = (fi(i, j) - fi(i - 1, j)) / dx;
//					v5 = (fi(i - 1, j) - fi(i - 2, j)) / dx;
//				}
//				S1 = 13.0 / 12 * pow(v1 - 2 * v2 + v3, 2) + 0.25 * pow(v1 - 4 * v2 + 3 * v3, 2);
//				S2 = 13.0 / 12 * pow(v2 - 2 * v3 + v4, 2) + 0.25 * pow(v2 - v4, 2);
//				S3 = 13.0 / 12 * pow(v3 - 2 * v4 + v5, 2) + 0.25 * pow(3 * v3 - 4 * v4 + v5, 2);
//
//				a1 = 0.1 / pow(e + S1, 2);
//				a2 = 0.6 / pow(e + S2, 2);
//				a3 = 0.3 / pow(e + S3, 2);
//
//				w1 = a1 / (a1 + a2 + a3);
//				w2 = a2 / (a1 + a2 + a3);
//				w3 = a3 / (a1 + a2 + a3);
//
//				f(i, j) = w1 * (v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6) + w2 * (-v2 / 6 + 5 * v3 / 6 + v4 / 3) + w3 * (v3 / 3 + 5 * v4 / 6 - v5 / 6);
//				f(i, j) = U(i, j) * f(i, j);
//			}
//	}
//	else {
//		for (int i = 3; i < ni - 3; i++)
//			for (int j = 3; j < nj - 3; j++)
//			{
//				double dy = ynode(j) - ynode(j - 1);
//				if (V(i, j) > 0.0) {
//					v1 = (fi(i, j - 2) - fi(i, j - 3)) / dy;
//					v2 = (fi(i, j - 1) - fi(i, j - 2)) / dy;
//					v3 = (fi(i, j) - fi(i, j - 1)) / dy;
//					v4 = (fi(i, j + 1) - fi(i, j)) / dy;
//					v5 = (fi(i, j + 2) - fi(i, j + 1)) / dy;
//				}
//				else {
//					v1 = (fi(i, j + 3) - fi(i, j + 2)) / dy;
//					v2 = (fi(i, j + 2) - fi(i, j + 1)) / dy;
//					v3 = (fi(i, j + 1) - fi(i, j)) / dy;
//					v4 = (fi(i, j) - fi(i, j - 1)) / dy;
//					v5 = (fi(i, j - 1) - fi(i, j - 2)) / dy;
//				}
//				S1 = 13.0 / 12 * pow(v1 - 2 * v2 + v3, 2) + 0.25 * pow(v1 - 4 * v2 + 3 * v3, 2);
//				S2 = 13.0 / 12 * pow(v2 - 2 * v3 + v4, 2) + 0.25 * pow(v2 - v4, 2);
//				S3 = 13.0 / 12 * pow(v3 - 2 * v4 + v5, 2) + 0.25 * pow(3 * v3 - 4 * v4 + v5, 2);
//
//				a1 = 0.1 / pow(e + S1, 2);
//				a2 = 0.6 / pow(e + S2, 2);
//				a3 = 0.3 / pow(e + S3, 2);
//
//				w1 = a1 / (a1 + a2 + a3);
//				w2 = a2 / (a1 + a2 + a3);
//				w3 = a3 / (a1 + a2 + a3);
//
//				f(i, j) = w1 * (v1 / 3 - 7 * v2 / 6 + 11 * v3 / 6) +
//					w2 * (-v2 / 6 + 5 * v3 / 6 + v4 / 3) +
//					w3 * (v3 / 3 + 5 * v4 / 6 - v5 / 6);
//				f(i, j) = V(i, j) * f(i, j);
//
//			}
//
//	}
//	return f;
// }

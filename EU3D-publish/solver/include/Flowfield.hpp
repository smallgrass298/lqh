/*=================================================================================
 * Class Flowfield:	Class for flow field processing
 *					Read file -> Construction -> Initialization ->
 *					Get the time step -> Set the boundary -> Update
 *=================================================================================*/

#ifndef FlowField
#define FlowField

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <iostream>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Array.hpp"
#include "TimeAdv.hpp"
#include "Reaction.hpp"
#include "Function.hpp"

using namespace std;
using namespace ARRAY;

class Flowfield
{
private:
	/* x axis */
	int ni;					/* Number of nodes */
	Array<double, 1> xnode; /* The mesh nodes */
	double dx;				/* The mesh spacing */
	/* y axis */
	int nj;					/* Number of nodes */
	Array<double, 1> ynode; /* The mesh nodes */
	double dy;				/* The mesh spacing */
	/* z axis */
	int nk;					/* Number of nodes */
	Array<double, 1> znode; /* The mesh nodes */
	double dz;				/* The mesh spacing */

	int bc; /* The number of each boundary grid */

	double dt;	 /* The time step */
	double time; /* The real time */

	double ft1 = 0, ft2 = 0, ft3 = 0, ft4 = 0, ft5 = 0, ft6 = 0;

	int NS;					  /* Number of species */
	int NR;					  /* Number of reactions */
	Reaction React;			  /* Class Reaction -> Object React */
	const double R = 8.31434; /* gas constant of ideal gas J/(mol*K) */
	Array<double, 1> Mw;	  /* Component molecular weight */
	Array<double, 1> Ri;	  /* Component gas constant */
	Array<double, 2> Coeff0;  /* NASA Thermochemical polynomial coefficient (T<1000) */
	Array<double, 2> Coeff1;  /* NASA Thermochemical polynomial coefficient (T>1000) */

	TimeAdv Time; /* Class TimeAdv -> Object Time */

	/*3D-RSBI*/
	double M_wave; /* Shockwave mach number for 3d-RSBI*/
	double U_post; /* Post-shock U velocity for 3d-RSBI */
	double P_pre;  /* Pre-shock pressure for 2d-RSBI */
	double P_post; /* Post-shock pressure for 2d-RSBI */
	double T_pre;  /* Pre-shock temperature for 2d-RSBI*/
	double T_post; /* Post-shock temperature for 2d-RSBI*/

	double cita; /* The wedge angle */
	double V3;
	Array<double, 3> U;		/* U velocity */
	Array<double, 3> V;		/* V velocity */
	Array<double, 3> W;		/* W velocity */
	Array<double, 3> P;		/* Pressure */
	double P_bound;			/* Boundary pressure for 1d-ZND problem */
	Array<double, 3> D;		/* Density */
	Array<double, 3> T;		/* Temperature */
	double T_bound;			/* Boundary temperature for 1d-ZND problem */
	Array<double, 3> C;		/* Sound speed */
	Array<double, 3> Ma;	/* Mach number */
	Array<double, 3> Wav;	/* Average molecule weight */
	Array<double, 3> Rgas;	/* Constant number of gas */
	Array<double, 3> Cp;	/* The specific heat at constant pressure */
	Array<double, 3> H;		/* Enthalpy */
	Array<double, 3> E;		/* Internal energy */
	Array<double, 3> Gamma; /* Specific heat ratio */
	Array<double, 4> Mr;	/* Component mole ratio */
	Array<double, 4> Mc;	/* Component mole concentration */
	Array<double, 4> Mi;	/* Component mole fraction */
	Array<double, 4> Yi;	/* Component mass fraction */
	Array<double, 4> Di;	/* Component density fraction */
	Array<double, 1> Cpi;	/* The specific heat at constant pressure of each component */
	Array<double, 1> Hi;	/* Enthalpy of each component */
	Array<double, 1> Ei;	/* Internal energy of each component */

	Array<double, 4> F;	 /* Flux at x direction */
	Array<double, 4> G;	 /* Flux at y direction */
	Array<double, 4> Q;	 /* Flux at z direction*/
	Array<double, 4> CS; /* Conserved variables */

	Function Fun; /* Class Function -> Object Fun */

	/* Freestream boundary */
	Array<double, 2> Uint, Vint, Wint, Pint, Dint, Tint, Hint, Eint, Gint;
	Array<double, 3> Yint;

	/* Intermediate variables */
	Array<double, 1> Yi_temp0, YL, YR;
	Array<double, 4> PLR, DLR, ULR, VLR, WLR, HLR, GLR, YLR, YL_temp, YR_temp;
	Array<double, 3> Yi_temp;
	Array<double, 4> Partial_T; /* Partial T */
	Array<double, 4> RHS;		/* Residual */

	/* Mpi variables */
	Array<double, 1> m_U_s, m_V_s, m_P_s, m_D_s, m_C_s, m_Gamma_s, m_H_s, m_T_s, m_Yi_s; /* Variables for send message */
	Array<double, 1> m_U_r, m_V_r, m_P_r, m_D_r, m_C_r, m_Gamma_r, m_H_r, m_T_r, m_Yi_r; /* Variables for recieve message */
	Array<double, 1> send_data_1, recv_data_1, send_data_2, recv_data_2, send_data_3, recv_data_3;

	struct SendRecv_Data
	{
		Array<double, 1> D, U, V, P, T, H, Gamma, C, Yi;
		void Iinitial(int size, int _NS)
		{
			D.Initial(size);
			U.Initial(size);
			V.Initial(size);
			P.Initial(size);
			T.Initial(size);
			H.Initial(size);
			Gamma.Initial(size);
			C.Initial(size);
			Yi.Initial(size * _NS);
		}
	};
	SendRecv_Data send_data, recv_data;

public:
	friend class Euler; /* Friend class -> access to use private object */

	/* Constructor */
	Flowfield() = default;

	/* Read the input file */
	void InputRead(char *initialization, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, int bc);

	/* Construct the flow field */
	void Construction();

	/* Reconstruct teh flow field for DLB */
	void ReConstruction(int meshnum);

	/* Initialize the flow field */
	void FieldInitial(Array<double, 1> &Ri, Array<double, 1> &Mw, Array<double, 4> &Mi_temp, Array<double, 4> &Yi_temp, Array<double, 2> &Coeff0, Array<double, 2> &Coeff1);

	/* Get the time step */
	void CFLcondition(double cfl, double Final_Time);

	/* Set the boundary condition */
	void FieldBoundary_3dODW();
	void FieldBoundary_1dZND(); /* For 1dZND */
	void FieldBoundary_3dRSBI();

	/* Solve the advection term */
	void Advection(int, int);

	/* Update after solving the advection term */
	void Update_after_Adv();

	/* Update the flow field after obtaining Di */
	void Explicit();
	void Explicit(int step, int i, int j, int k);

	/* Update the flow field */
	void Update_IMEX(Array<double, 4> &Wi, Array<double, 5> &MD);

	/* Update after the Conserved variables */
	void Update_after_CS();
	void Update_after_CS(int step, int i, int j, int k);

	/* Destructor */
	~Flowfield() { ; };

	/* AUSM */
	void AUSM(int, Array<double, 4> &, void (*Diff)(int, Array<double, 1> &, Array<double, 1> &, Array<double, 1> &, Array<double, 4> &, Array<double, 3> &, int, int));

	/* Get the temperature by Newton iteration */
	double Get_temp(double, int, int, int);
	double Get_temp(double, int, int, int, Array<double, 1> &);

	/* Get the Partial_T */
	void GetPartial_T();

	/* Mpi process */
	void Mpi_Boundary();
	// void Mpi_Process1();
	// void Mpi_Process2();

	/* Package transfer flowfield data */
	void PackagePrev(std::vector<double> &, int i, int j, int k);
	void PackageUpdate(std::vector<double> &, int meshnum);
	void PackageUpdate(std::vector<double> &, int, std::vector<int> &, std::vector<int> &);
	void PackageUpdate(std::vector<double> &, std::vector<int> &, std::vector<int> &);

	/* Unpackage transfer flowfield data */
	void UnpackagePrev(std::vector<double> &, int meshnum);
	void UnpackagePrev(std::vector<double> &, int, std::vector<int> &, std::vector<int> &);
	void UnpackagePrev(std::vector<double> &, std::vector<int> &, std::vector<int> &);
	void UnpackagePrev(std::vector<double> &, Array<int, 1> &, Array<int, 1> &);
	void UnpackageUpdate(std::vector<double> &, std::vector<int> &, int, std::vector<int> &, std::vector<int> &);
	void UnpackageUpdate(std::vector<double> &, std::vector<int> &, std::vector<int> &, std::vector<int> &);
};
#endif
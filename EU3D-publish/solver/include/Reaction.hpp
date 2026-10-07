/*=================================================================================
 * Class Reaction:	Class for reaction process
 *					Read file -> Construction -> Initialization -> Reaction
 *=================================================================================*/

#ifndef REACTION_H
#define REACTION_H

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Array.hpp"
using namespace ARRAY;

class Reaction
{
private:
	/* x axis */
	int ni;					/* Number of nodes */
	Array<double, 1> xnode; /* The mesh nodes */
	/* y axis */
	int nj;					/* Number of nodes */
	Array<double, 1> ynode; /* The mesh nodes */
	/* z axis */
	int nk;					/* Number of nodes */
	Array<double, 1> znode; /* The mesh nodes */

	int bc; /* The number of each boundary grid */

	int NS; /* Number of species */
	int NR; /* Number of reactions */
	int TB; /* Number of three-body reactions */

	Array<double, 2> Stoi_F; /* Stoichiometric coefficient of reactants */
	Array<double, 2> Stoi_B; /* Stoichiometric coefficient of products */

	/* Forward reactions */
	Array<double, 1> Af;  /* Pre-exponent factor */
	Array<double, 1> Bf;  /* Temperature index */
	Array<double, 1> Eaf; /* Activation energy */

	Array<double, 2> React_TB; /* Coefficient of three-body reactions */

	Array<double, 2> Coeff0; /* NASA Thermochemical polynomial coefficient (T<1000) */
	Array<double, 2> Coeff1; /* NASA Thermochemical polynomial coefficient (T>1000) */

	const double R = 8.31434; /* gas constant of ideal gas J/(mol*K) */
	const double Ru = 1.987;  /* gas constant of ideal gas cal/(mol*K) */
	const double P0 = 101325; /* one standard atmosphere */

	Array<double, 1> Mw; /* Component molecular weight */
	Array<double, 1> Ri; /* Component gas constant */

	Array<double, 4> Mc;	  /* Component mole concentration */
	Array<double, 4> Mr;	  /* Component mole ratio */
	Array<double, 1> Mr_temp; /* Component mole ratio */
	Array<double, 4> Mi;	  /* Component mole fraction */
	Array<double, 4> Yi;	  /* Component mass fraction */
	Array<double, 4> Di;	  /* Component density fraction */

	double T_local; /* Local temperature */

	Array<double, 1> Hi; /* Enthalpy of each component */
	Array<double, 1> Si; /* Entropy of each component */
	Array<double, 1> Gi; /* Gibbs free energy of each component */

	Array<double, 1> KF; /* Forward reaction rate constant */
	Array<double, 1> KB; /* Reverse reaction rate constant */
	Array<double, 1> Kp; /* Chemical reaction equilibrium constant expressed in terms of pressure */
	Array<double, 1> Kc; /* Chemical reaction equilibrium constant expressed in terms of concentration */

	Array<double, 1> RR_F; /* Forward reaction rate */
	Array<double, 1> RR_B; /* Reverse reaction rate */
	Array<double, 1> R_TB; /* Correction coefficient of three-body reactions */
	Array<double, 1> RR;   /* Net reaction rate */

	Array<double, 1> Wi;		 /* Component mass production rate */
	Array<double, 4> CMS;		 /* Component mass production rate */
	Array<double, 2> WJH1, WJH2; /* Intermediate variables */
	Array<double, 5> MD;		 /* Diagonal of the Jacobi matrix */

	Array<double, 1> P, Q; /* The variables for Trapezoid formula */

	Array<int, 3> Nchem;			/* The number of chemical iteration step of every mesh node */
	int NchemNow = 1;				/* The current maximum Nchem */
	std::vector<int> NchemMax_Rank; /* Restore current maximum Nchem for each iteration of each process */
	Array<int, 1> NchemMax_Total;	/* Restore maximum Nchem of whole computation domain */
	std::vector<int> NchemMax;		/* Restore maximum Nchem */
	int NchemSum;					/* Restore total Nchem of each process */
	Array<int, 1> NchemTotal;		/* Restore total Nchem of all process and corresponding process id */
Array<int, 1> NchemTotalBalance;;
	struct intermidatePara
	{
		double P[16];
		double Q[16];
		double Hi[16];
		double Si[16];
		double Gi[16];
		double Wi[16];
		double RR_F[24];
		double RR_B[24];
		double R_TB[24];
		double KF[24];
		double Kp[24];
		double Kc[24];
		double KB[24];
		double RR[24];
	};
	struct interParaDiag
	{
		double P[16];
		double Q[16];
		double Hi[16];
		double Si[16];
		double Gi[16];
		double Wi[16];
		double RR_F[24];
		double RR_B[24];
		double R_TB[24];
		double KF[24];
		double Kp[24];
		double Kc[24];
		double KB[24];
		double RR[24];
		double WJH1[12][24] = {};
		double WJH2[12][24] = {};
	};

public:
	friend class Euler; /* Friend class -> access to use private object */

	/* Constructor */
	Reaction() = default;

	/* Read the reaction model file and thermo properties */
	void ReactionRead(char *reaction_model, char *thermofile, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, int bc);

	/* Construction */
	void ReactionConstruction();

	void ReConstruction(int meshnum);

	/* Calculate some initial parameters */
	void ReactionInitial();

	/* Reaction process using Trapezoid formula */
	Array<double, 4> Trapezoid(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 3> &T, double dt);
	void Trapezoid(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 4> &Yi_temp, Array<double, 3> &T, double dtm, int step, int i, int j, int k);
	void TrapezoidPrediction(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 4> &Yi_temp, Array<double, 3> &T, double dtm);

	/* Reaction process using IMEX with diagonalized matrix */
	void Diagonalized(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 3> &T, Array<double, 4> &Partial_T);
	/* Reaction process using Cantera or DNN */
	// Array<double> ReactionS(int type, Array<double> Di, Array<double> T, Array<double> P, double dt);

	/* Destructor */
	~Reaction() { ; };

	/* Get the specific heat at constant pressure */
	double GetCpi(double T, double R, int SP, Array<double, 2> &, Array<double, 2> &);

	/* Get the enthalpy */
	double GetHi(double T, double R, int SP, Array<double, 2> &, Array<double, 2> &);

	/* Get the entropy */
	double GetSi(double T, double R, int SP, Array<double, 2> &, Array<double, 2> &);

	/* Get the component mass production rate */
	Array<double, 4> GetWi();

	/* Get the diagonal of the Jacobi matrix */
	Array<double, 5> GetMD();

	/* Push the maximum Nchem for whole computation domain */
	void PushNchemMax();
	/* Get the maximum Nchem for whole computation domain */
	void GetNchemMax();
	/* Restore transfer mesh data */
	void PackagePrev(std::vector<int> &, int i, int j, int k);
	void UnpackagePrev(std::vector<int> &, int, std::vector<int> &, std::vector<int> &);
	void UnpackagePrev(std::vector<int> &, std::vector<int> &, std::vector<int> &);
	void PackageUpdate(std::vector<int> &, int, std::vector<int> &, std::vector<int> &);
	void UnpackageUpdate(std::vector<int> &, std::vector<int> &, int, std::vector<int> &, std::vector<int> &);
};
#endif

/*=================================================================================
 * Class Euler:	The main solver
 *				Intialization -> Compute and Update -> Output
 *=================================================================================*/

#ifndef EULER_H
#define EULER_H

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <fstream>
#include <iostream>
#include <chrono>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Array.hpp"
#include "Mesh.hpp"
#include "Flowfield.hpp"
#include "Reaction.hpp"
#include "Global.hpp"
#include "LoadBalance.hpp"

using namespace ARRAY;
class Euler
{
private:
	char ctrlfile[100] = "./input/ctrl.txt"; /* Ctrl file */
	char meshFile[100];						 /* Grid file */
	char inputFile[100];					 /* Input file */
	char Reaction_Model[100];				 /* Reaction model file */
	char thermoFile[100];					 /* Thermo data file */
	char react[50] = {0}, diff[50] = {0}, timeadv[50] = {0};
	int Reaction_Sch;  /* The reaction scheme */
	int type;		   /* The type of selected reaction scheme */
	int Diff_Sch;	   /* The difference scheme */
	int TimeAdv_Sch;   /* The time advance scheme */
	double Final_Time; /* Final time */
	double cfl;		   /* The cfl number */
	double dt;		   /* The fixed time step (if necessary) */
	int count;		   /* The number of output files */
	int iteration;	   /* Iteration steps */

	double tr1 = 0, tr2 = 0, tr3 = 0, tr4 = 0;
	double dt1 = 0.0, dt2 = 0.0, dt3 = 0.0, dt4 = 0.0,dt5 = 0.0,dt6 = 0.0,dt7 = 0.0,dt8 = 0.0;
	double trackt=0.0;

	Mesh Mymesh;	  /* Class Mesh -> Object Mymesh */
	Flowfield TwoDim; /* Class Flowfield -> Object TwoDim */
	Reaction React;	  /* Class Reaction -> Object React */

	/* DLB */
	Flowfield ExtraFlow;
	Reaction ExtraReact;
	DynamicLoadBalancer DLB;
	int DLB_step = 0;
	double DLB_tol = 0;
	int pos = 0;
	int TransferNchem = 0;
	int AverageNchem = 0;
	Array<int, 1> NchemIndex;
	std::vector<int> TransferIndex;
	int SendTransferMeshNum = 0;
	int RecvTransferMeshNum = 0;
	std::vector<double> PrevFlowData;
	std::vector<double> PrevFlowDataLocal;
	std::vector<double> SendPrevFlowDataLocal;
	std::vector<double> RecvPrevFlowDataLocal;
	std::vector<double> UpdateFlowData;
	std::vector<double> UpdateFlowDataLocal;
	std::vector<double> SendUpdateFlowDataLocal;
	std::vector<double> RecvUpdateFlowDataLocal;
	std::vector<double> PrevFlowNumID;
	std::vector<double> UpdateFlowNumID;
	Array<int, 1> SendRecvNchem;
	Array<int, 1> SendRecvMeshNumLocal;
	Array<int, 1> SendRecvMeshNum;
	Array<int, 1> SendRecvMeshNumTemp;
	Array<int, 1> RecvPrevFlowMeshNum;
	Array<int, 1> GatherRecvCounts;
	Array<int, 1> GatherDispls;
	int sendid = 0;
	int recvid = 0;
	Array<int, 1> Capacity;
	int Capacity_local = 0;
	MPI_Request requests1[2];
	MPI_Status statuses1[2];
	MPI_Request requests2[2];
	MPI_Status statuses2[2];

	/* Read the file */
	void
	FileRead();

public:
	/* Constructor */
	Euler();

	/* MPI intilization */
	void Mpi_Initial();

	/* Solve the Euler equation and update the flow field */
	void Computing(int count);

	/* GPU time loop with periodic host output. */
	void Computing_GPU(int count);

	/* Solve the reaction term by trapezoid formula */
	void Trapezoid(int type);

	/* Solve the reaction term by IMEX method */
	void IMEX();

	/* Solve the reaction term by Cantera or DNN method */
	void DNN(int type);

	/* Output the result in file */
	void Output(int num);

	/* Output the time used of each function block */
	void Output_Total_Time(double t1, double t2, double t3, double t4, double t5, double t6, double t7, double t8, double t9, double t10);
	void Output_Compute_Time(double t1, double t2, double t3, double t4, double t5);
	/* Destructor */
	~Euler()
	{
		;
	};
};
#endif

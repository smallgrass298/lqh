/*=================================================================================
 * The functions of Euler
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <cstring>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mpi.h>
#include <vector>
#include <numeric>
#include <filesystem>
#include <cstdlib>
#include <cmath>
#include <iomanip>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Euler.hpp"
#include "_string_switch.hpp"
#include "FileReader.hpp"

using namespace std;

/*---------------------------------------------------------------------------------
	Name:		Euler::Euler

	Input(0):	None

	Function:	Read files which identify the physical problem;
				Initialization;
				Compute the flow field and output in real time;
				Output the final result

	Return:		None
 *---------------------------------------------------------------------------------*/

Euler::Euler()
{
	if (const char *ctrl_env = std::getenv("EU3D_CTRL"))
	{
		std::strncpy(ctrlfile, ctrl_env, sizeof(ctrlfile) - 1);
		ctrlfile[sizeof(ctrlfile) - 1] = '\0';
	}
	/* Print the result of pre-condition */
	cout << "Input ctrl file is:" << this->ctrlfile << endl;

	/* Read the ctrl file */
	double t1 = MPI_Wtime();
	FileRead();
	double t2 = MPI_Wtime();
	/* Mpi initialization: calculate neighbor process id*/
	Mpi_Initial();
	double t3 = MPI_Wtime();
	/* Read the mesh file and process the mesh */
	Mymesh.MeshProcess(meshFile);
	double t4 = MPI_Wtime();
	/* Read the input file */
	TwoDim.InputRead(inputFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	ExtraFlow.InputRead(inputFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	double t5 = MPI_Wtime();
	/* Read the reaction model and thermo file */
	React.ReactionRead(Reaction_Model, thermoFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	ExtraReact.ReactionRead(Reaction_Model, thermoFile, Mymesh.xnode, Mymesh.ynode, Mymesh.znode, Mymesh.bc);
	double t6 = MPI_Wtime();
	/* Intitalize the flow field */
	TwoDim.FieldInitial(React.Ri, React.Mw, React.Mi, React.Yi, React.Coeff0, React.Coeff1);
	ExtraFlow.FieldInitial(React.Ri, React.Mw, React.Mi, React.Yi, React.Coeff0, React.Coeff1);
	double t7 = MPI_Wtime();
	MPI_Barrier(MPI_COMM_WORLD);
	double t8 = MPI_Wtime();
	/* Compute the flow field and output in real time */
#ifdef USE_GPU
	Computing_GPU(count);   /* Run the main loop on the GPU. */
#else
	Computing(count);
#endif
	double t9 = MPI_Wtime();
	/* Output the final result in a file */
	Output(count);
	double t10 = MPI_Wtime();
	Output_Total_Time(t1, t2, t3, t4, t5, t6, t6, t8, t9, t10);
	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::FileRead

	Input(0):	None

	Function:	Read the ctrl file to get the global informartion

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::FileRead()
{
	FileReader Ctrl;
	Ctrl.readFile(ctrlfile);

	/* Read number */
	Final_Time = Ctrl.getdoubleParameter("Final-Time");
	cfl = Ctrl.getdoubleParameter("cfl");
	if (cfl == 0)
		dt = Ctrl.getdoubleParameter("dt");
	count = Ctrl.getIntParameter("count");

	/* Read string */
	strcpy(meshFile, Ctrl.getStringParameter("gridname").c_str());
	strcpy(inputFile, Ctrl.getStringParameter("initialization").c_str());
	strcpy(Reaction_Model, Ctrl.getStringParameter("reaction-model").c_str());
	strcpy(thermoFile, Ctrl.getStringParameter("Thermofile").c_str());

	/* Judge the reaction scheme */
	strcpy(react, Ctrl.getStringParameter("reaction").c_str());
	switch (hash_(react))
	{
	case "Trapezoid"_hash:
		Reaction_Sch = Global::Trapezoid;
		type = 0;
		break;
	case "Trapezoid-Adaptive"_hash:
		Reaction_Sch = Global::Trapezoid;
		type = 1;
		break;
	case "IMEX"_hash:
		Reaction_Sch = Global::IMEX;
		break;
	case "Cantera"_hash:
		Reaction_Sch = Global::DNN;
		type = 0;
		break;
	case "DNN"_hash:
		Reaction_Sch = Global::DNN;
		type = 1;
		break;
	default:
		cout << "Invalid reaction scheme" << endl;
		abort();
	}

	/* Judge the difference scheme */
	strcpy(diff, Ctrl.getStringParameter("difference").c_str());
	switch (hash_(diff))
	{
	case "MUSCL_1"_hash:
		Diff_Sch = Global::MUSCL_1;
		break;
	case "MUSCL_2"_hash:
		Diff_Sch = Global::MUSCL_2;
		break;
	case "WENO"_hash:
		Diff_Sch = Global::WENO;
		break;
	default:
		cout << "Invalid difference scheme" << endl;
	}

	/* Judge the time advance scheme */
	strcpy(timeadv, Ctrl.getStringParameter("time-adv").c_str());
	switch (hash_(timeadv))
	{
	case "EE"_hash:
		TimeAdv_Sch = Global::EE;
		break;
	case "TVD_RK3"_hash:
		TimeAdv_Sch = Global::TVD_RK3;
		break;
	default:
		cout << "Invalid time-advance scheme" << endl;
	}

	/* Read mpi block number */
	m_block_x = Ctrl.getIntParameter("m_block_x");
	m_block_y = Ctrl.getIntParameter("m_block_y");
	m_block_z = Ctrl.getIntParameter("m_block_z");

	/* Read openmp thread number */
	num_thread = Ctrl.getIntParameter("num_thread");

	/* Read DLB parameters */
	DLB_step = Ctrl.getIntParameter("DLB_step");
	DLB_tol = Ctrl.getdoubleParameter("DLB_tol");

	/* Create output directory based on process number */
	if (myid == 0)
	{
		std::string outputFolder = "./output_" + std::to_string(numprocs);
		std::filesystem::create_directory(outputFolder);

		std::filesystem::create_directory(outputFolder + "/grid");
		std::filesystem::create_directory(outputFolder + "/load");
		std::filesystem::create_directory(outputFolder + "/time_record");
		std::filesystem::create_directory(outputFolder + "/time_record/all");
		std::filesystem::create_directory(outputFolder + "/time_record/compare");
		std::filesystem::create_directory(outputFolder + "/time_record/compute");
		std::filesystem::create_directory(outputFolder + "/time_record/total");
		std::filesystem::create_directory(outputFolder + "/results");

		std::cout << "目录结构创建成功: " << outputFolder << std::endl;
	}

	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Mpi_Initial

	Input(0):	None

	Function:	Initialize mpi information

	Return:		None
 *---------------------------------------------------------------------------------*/
void Euler::Mpi_Initial()
{
	const int configured_procs = m_block_x * m_block_y * m_block_z;
	if (configured_procs != numprocs)
	{
		if (myid == 0)
			cerr << "MPI process mismatch: ctrl.txt requests " << configured_procs
				 << " processes (" << m_block_x << "x" << m_block_y << "x" << m_block_z
				 << "), but mpirun started " << numprocs << endl;
		MPI_Abort(MPI_COMM_WORLD, 2);
	}
	myid_x = int(myid % m_block_x);
	myid_z = int(floor(myid / (m_block_x * m_block_y)));
	myid_y = int(floor(myid % (m_block_x * m_block_y) / m_block_x));
	if (myid_x > 0)
		m_left = myid - 1;
	else
		m_left = MPI_PROC_NULL;
	if (myid_x < m_block_x - 1)
		m_right = myid + 1;
	else
		m_right = MPI_PROC_NULL;

	if (myid_y > 0)
		m_front = myid - m_block_x;
	else
		m_front = MPI_PROC_NULL;
	if (myid_y < m_block_y - 1)
		m_back = myid + m_block_x;
	else
		m_back = MPI_PROC_NULL;

	if (myid_z > 0)
		m_down = myid - m_block_x * m_block_y;
	else
		m_down = MPI_PROC_NULL;
	if (myid_z < m_block_z - 1)
		m_up = myid + m_block_x * m_block_y;
	else
		m_up = MPI_PROC_NULL;
	// cout << myid << " " << myid_x << " " << myid_y << " " << myid_z << " " << m_left << " " << m_right << " " << m_front << " " << m_back << " " << m_down << " " << m_up << endl;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Computing

	Input(1):	The number of output files

	Function:	Solve and update the flow field;
				Output the result in real time

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Computing(int count)
{
	double time = 0.0;
	int num = 0;
	iteration = 0;
	/* Fixed-dt cases must execute an exact integer number of steps.  Comparing
	 * accumulated floating-point time against Final_Time can otherwise run one
	 * extra step (e.g. 3e-7 / 3e-8). */
	const long long fixed_steps = (cfl == 0.0)
		? static_cast<long long>(std::ceil(Final_Time / dt - 1.0e-12)) : -1;
	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
	double t12 = 0, t23 = 0, t34 = 0, t45 = 0, t67 = 0;
	double tt1 = 0, tt2 = 0;
	ofstream loadfile;
	if (myid == 0)
		loadfile.open("./output_" + to_string(numprocs) + "/load/LoadDegree.dat");
	tt1 = MPI_Wtime();
	while ((cfl == 0.0 && iteration < fixed_steps) ||
		   (cfl != 0.0 && time < Final_Time))
	{
		/* Calculate the time step */
		if (cfl == 0)
			TwoDim.dt = dt;
		else
			TwoDim.CFLcondition(cfl, Final_Time);

		/* Ouput the real-time result unifomrly */
		if (time <= num * Final_Time / count && time + TwoDim.dt > num * Final_Time / count)
		{
			t6 = MPI_Wtime();
			Output(num);
			num = num + 1;
			t7 = MPI_Wtime();
			t67 += t7 - t6;
		}
		t1 = MPI_Wtime();
		/* Update the boundary */
		#ifdef EU3D_DEBUG_DUMPS
		Output(10);
		#endif
		TwoDim.FieldBoundary_3dRSBI();
		#ifdef EU3D_DEBUG_DUMPS
		Output(11);
		#endif
		t2 = MPI_Wtime();
		t12 += t2 - t1;
		TwoDim.Mpi_Boundary();
		#ifdef EU3D_DEBUG_DUMPS
		Output(12);
		#endif
		t3 = MPI_Wtime();
		t23 += t3 - t2;
		/* Solve the advection term */
		TwoDim.Advection(TimeAdv_Sch, Diff_Sch);
		#ifdef EU3D_DEBUG_DUMPS
		Output(13);
		#endif
		// TwoDim.Update_after_Adv();
		t4 = MPI_Wtime();
		t34 += t4 - t3;
		/* Identify the reaction scheme */
		switch (Reaction_Sch)
		{
		case 0:
			Trapezoid(type);
			break;
		case 1:
			IMEX();
			break;
		case 2:
			DNN(type);
			break;
		}
		t5 = MPI_Wtime();
		#ifdef EU3D_DEBUG_DUMPS
		Output(14);
		#endif
		t45 += t5 - t4;
		/* Monitor the iteration */
		time += TwoDim.dt;
		iteration += 1;
		if (myid == 0)
			cout << "Process " << myid << ":  Iteration:  " << iteration << "  " << TwoDim.dt << "  " << time << endl;

		/* Judge if there exists nan value */
		if (TwoDim.D.IsNan())
		{
			cout << "D is nan\n";
			Output(999);
			abort();
		}

		if (TwoDim.T.IsNan())
		{
			cout << "T is nan\n";
			Output(999);
			abort();
		}
		if (myid == 0)
			loadfile << iteration << '\t' << DLB.LoadDegree << '\t' << DLB.LoadDegreeBalance << '\n';
	}
	MPI_Barrier(MPI_COMM_WORLD);
	tt2 = MPI_Wtime();
	trackt = tt2 - tt1;
	Output_Compute_Time(t12, t23, t34, t45, t67);

	/* For adaptive-trapezoid method, output NchemMax */
	if (Reaction_Sch == 0 && type == 1)
		React.GetNchemMax();

	return;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::Trapezoid

	Input(1):	The type of the Trapezoid formula

	Function:	Solve the reaction term by trapezoid formula
				(Adaptive (1) or not (0))

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Trapezoid(int type)
{
	int ni = Mymesh.ni;
	int nj = Mymesh.nj;
	int nk = Mymesh.nk;
	int bc = Mymesh.bc;

	double t1 = 0, t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0;
	double tw1 = 0, tw2 = 0, tw3 = 0, tw4 = 0, tw5 = 0, tw6 = 0, tw7 = 0; // wait time record
	/* Update the flow field after solving the advection term */
	t1 = MPI_Wtime();

	TwoDim.Update_after_Adv();
	t2 = MPI_Wtime();
	tr1 += t2 - t1;
	if (type == 0)
	{
		/* Solve the reaction source term */
		TwoDim.Di = React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.T, TwoDim.dt);

		/* Update the flow field */
		TwoDim.Explicit();
	}
	else if (type == 1) /* Adaptive */
	{
		double ttp1 = MPI_Wtime();
		React.TrapezoidPrediction(TwoDim.Mc, TwoDim.Di, TwoDim.Yi, TwoDim.T, TwoDim.dt);
		double ttp2 = MPI_Wtime();
		dt4 += ttp2 - ttp1;
		/* Dynamic Load Balance */
		int nchemSumBalance = 0;
		/* Reassign work load */
		if (iteration % DLB_step == 0)
		{
			DLB.NchemSumBalance = 0;
			React.NchemSum = React.Nchem.SumNoBoundary(bc);
			MPI_Allgather(&React.NchemSum, 1, MPI_INT, &React.NchemTotal(0), 1, MPI_INT, MPI_COMM_WORLD);
			DLB.LoadDegree = double((React.NchemTotal.MaxValue() - React.NchemTotal.AveValue())) / React.NchemTotal.MaxValue();
			if (myid == 0)
				cout << "Load Degree: " << DLB.LoadDegree << endl;
			if (DLB.LoadDegree > DLB_tol)
			{
				// if (myid == 0)
				// {
				// cout << "Befor DLB: ";
				// React.NchemTotal.Print();
				// }
				DLB.transferNchem.clear();
				/* get the transfer records */
				DLB.DLBPriorityQueue(React.NchemTotal);
				DLB.transferMeshNum.resize(DLB.transferNchem.size());
				DLB.transferNmesh.resize(DLB.transferNchem.size());
				std::fill(DLB.transferMeshNum.begin(), DLB.transferMeshNum.end(), 0);
				std::fill(DLB.transferNmesh.begin(), DLB.transferNmesh.end(), 0);
				DLB.transferPrevDataLocal.clear();
				DLB.transferPrevData.clear();
				DLB.transferPrevNchemLocal.clear();
				DLB.transferPrevNchem.clear();
				DLB.transferIndex.clear();
				DLB.sentNmesh = 0;
				DLB.prevFlowDataLocal.clear();
				DLB.prevReactDataLocal.clear();
			}
		}
		MPI_Request sendRequests1[DLB.transferNchem.size()];
		MPI_Request sendRequests2[DLB.transferNchem.size()];
		MPI_Request sendRequests3[DLB.transferNchem.size()];
		MPI_Request recvRequests1[DLB.transferNchem.size()];
		MPI_Request recvRequests2[DLB.transferNchem.size()];
		MPI_Request recvRequests3[DLB.transferNchem.size()];
		MPI_Status statuses[DLB.transferNchem.size()];
		for (int i = 0; i < DLB.transferNchem.size(); i++)
		{
			sendRequests1[i] = MPI_REQUEST_NULL;
			sendRequests2[i] = MPI_REQUEST_NULL;
			sendRequests3[i] = MPI_REQUEST_NULL;
			recvRequests1[i] = MPI_REQUEST_NULL;
			recvRequests2[i] = MPI_REQUEST_NULL;
			recvRequests3[i] = MPI_REQUEST_NULL;
		}
		if (DLB.LoadDegree > DLB_tol)
		{
			double tt1 = MPI_Wtime();
			/* send myid judge transferIndex, package data, transfer mesh number */
			int sentNchem = DLB.GetSentNchem(myid);
			// cout << "\tmyid " << myid << " Nchem: " << sentNchem << endl;
			if (sentNchem != 0)
			{
				DLB.prevFlowDataLocal.clear();
				DLB.prevReactDataLocal.clear();

				std::vector<int> sendToTemp = DLB.sendTo;
				int nchem_temp = 0, id = 0, start = 0, n = 0;

				/* Get the transfer mesh number & transferIndex & package data */
				if (iteration % DLB_step == 0)
				{
					DLB.GetTransferMesh(React.Nchem, bc);
					DLB.sentNmesh = std::accumulate(DLB.transferMeshNum.begin(), DLB.transferMeshNum.end(), 0);
				}
				int x = 0, y = 0, z = 0;
				for (int i = 0; i < DLB.transferIndex.size() / 3; i++)
				{
					x = DLB.transferIndex[3 * i];
					y = DLB.transferIndex[3 * i + 1];
					z = DLB.transferIndex[3 * i + 2];
					TwoDim.PackagePrev(DLB.prevFlowDataLocal, x, y, z);
					React.PackagePrev(DLB.prevReactDataLocal, x, y, z);
				}
			}
			double tt2 = MPI_Wtime();
			dt1 += tt2 - tt1;
			if (iteration % DLB_step == 0)
			{ /* reduce transfer mesh number to 0 process */
				MPI_Reduce(DLB.transferMeshNum.data(), DLB.transferNmesh.data(), DLB.transferMeshNum.size(), MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
				MPI_Bcast(DLB.transferNmesh.data(), DLB.transferNmesh.size(), MPI_INT, 0, MPI_COMM_WORLD);
				if (myid == 0)
				{
					cout << "TransferNmesh: " << std::accumulate(DLB.transferNmesh.begin(), DLB.transferNmesh.end(), 0) << endl;
				}
			}
			double ttt1 = MPI_Wtime();
			if (sentNchem != 0)
				for (int i = 0; i < DLB.sendTo.size(); i++)
				{
					int start1 = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0) * (5 + 3 * TwoDim.NS);
					int start2 = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0);
					int size1 = DLB.transferNmesh[DLB.sendTo[i]] * (5 + 3 * TwoDim.NS);
					int size2 = DLB.transferNmesh[DLB.sendTo[i]];
					if (size2 != 0)
					{
						MPI_Isend(&DLB.prevFlowDataLocal[start1], size1, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i], MPI_COMM_WORLD, &sendRequests1[DLB.sendTo[i]]);
						MPI_Isend(&DLB.prevReactDataLocal[start2], size2, MPI_INT, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i] + 10000, MPI_COMM_WORLD, &sendRequests2[DLB.sendTo[i]]);
					}
				}
			double ttt2 = MPI_Wtime();
			int recvNmesh = DLB.GetRecvNmesh(myid);
			/* Unpackage update data */
			if (recvNmesh != 0)
			{
				DLB.recvPrevFlowData.clear();
				DLB.recvPrevReactData.clear();
				DLB.recvPrevFlowData.resize(recvNmesh * (5 + 3 * TwoDim.NS));
				DLB.recvPrevReactData.resize(recvNmesh);
				for (int i = 0; i < DLB.recvFrom.size(); i++)
				{
					int start1 = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0) * (5 + 3 * TwoDim.NS);
					int start2 = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0);
					int size1 = DLB.transferNmesh[DLB.recvFrom[i]] * (5 + 3 * TwoDim.NS);
					int size2 = DLB.transferNmesh[DLB.recvFrom[i]];
					if (size2 != 0)
					{
						// MPI_Recv(&DLB.recvPrevFlowData[start1], size1, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
						// 		 DLB.recvFrom[i], MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						// MPI_Recv(&DLB.recvPrevReactData[start2], size2, MPI_INT, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
						// 		 DLB.recvFrom[i] + 10000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Irecv(&DLB.recvPrevFlowData[start1], size1, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i], MPI_COMM_WORLD, &recvRequests1[DLB.recvFrom[i]]);
						MPI_Irecv(&DLB.recvPrevReactData[start2], size2, MPI_INT, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i] + 10000, MPI_COMM_WORLD, &recvRequests2[DLB.recvFrom[i]]);
					}
				}
			}
		}
		int tag = 0;
		nchemSumBalance = 0;
		// #pragma omp parallel for num_threads(num_thread) collapse(3) schedule(dynamic, 100) reduction(+ : nchemSumBalance)
		for (int i = bc; i < (ni - 1) / 2 + bc; i++)
			for (int j = bc; j < nj + bc; j++)
				for (int k = bc; k < nk + bc; k++)
				{
					int n = 0;
					bool flag = false;
					while (n < DLB.sentNmesh && tag < DLB.sentNmesh)
					{
						if (i == DLB.transferIndex[3 * n] &&
							j == DLB.transferIndex[3 * n + 1] &&
							k == DLB.transferIndex[3 * n + 2])
						{
							flag = true;
							tag++;
							break;
						}
						n++;
					}
					if (!flag)
					{
						nchemSumBalance += React.Nchem(i, j, k);
						for (int step = 0; step < React.Nchem(i, j, k); step++)
						{
							/* Solve the reaction source term */
							React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.Yi, TwoDim.T, TwoDim.dt, step, i, j, k);

							/* Update the flow field */
							TwoDim.Explicit(step, i, j, k);
						}
					}
				}
		DLB.NchemSumBalance += nchemSumBalance;

		tw1 = MPI_Wtime();
		MPI_Waitall(DLB.transferNchem.size(), sendRequests1, statuses);
		MPI_Waitall(DLB.transferNchem.size(), sendRequests2, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests1, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests2, statuses);
		tw2 = MPI_Wtime();

		if (DLB.LoadDegree > DLB_tol)
		{
			int recvNmesh = DLB.GetRecvNmesh(myid);
			/* Unpack transfer data */
			if (recvNmesh != 0)
			{
				nchemSumBalance = 0;
				ExtraFlow.ReConstruction(recvNmesh);
				ExtraFlow.UnpackagePrev(DLB.recvPrevFlowData, DLB.recvFrom, DLB.transferNmesh);
				ExtraReact.ReConstruction(recvNmesh);
				ExtraReact.UnpackagePrev(DLB.recvPrevReactData, DLB.recvFrom, DLB.transferNmesh);
				// cout << myid << " Start DLB\n";

				// #pragma omp parallel for num_threads(num_thread) collapse(1) schedule(dynamic) reduction(+ : nchemSumBalance)
				for (int i = 0; i < recvNmesh; i++)
				{
					// DLB.NchemSumBalance += ExtraReact.Nchem(i, 0, 0);
					nchemSumBalance += ExtraReact.Nchem(i, 0, 0);
					for (int step = 0; step < ExtraReact.Nchem(i, 0, 0); step++)
					{
						/* Solve the reaction source term */
						ExtraReact.Trapezoid(ExtraFlow.Mc, ExtraFlow.Di, ExtraFlow.Yi, ExtraFlow.T, TwoDim.dt, step, i, 0, 0);
						/* Update the flow field */
						ExtraFlow.Explicit(step, i, 0, 0);
					}
				}
				DLB.NchemSumBalance += nchemSumBalance;
				double tt4 = MPI_Wtime();
				// dt3 += tt4 - tt3;
				DLB.updateFlowDataLocal.clear();
				DLB.updateFlowDataLocal.resize(recvNmesh * (18 + 4 * TwoDim.NS));
				ExtraFlow.PackageUpdate(DLB.updateFlowDataLocal, DLB.recvFrom, DLB.transferNmesh);
				bool has_nan = std::any_of(DLB.updateFlowDataLocal.begin(), DLB.updateFlowDataLocal.end(), [](double x)
										   { return std::isnan(x); });
				if (has_nan)
				{
					std::cout << "DLB contains NaN." << std::endl;
					abort();
				}
				for (int i = 0; i < DLB.recvFrom.size(); i++)
				{
					int start = std::accumulate(DLB.transferNmesh.begin() + DLB.recvFrom[0], DLB.transferNmesh.begin() + DLB.recvFrom[i], 0) * (18 + 4 * TwoDim.NS);
					int size = DLB.transferNmesh[DLB.recvFrom[i]] * (18 + 4 * TwoDim.NS);
					if (size != 0)
						MPI_Isend(&DLB.updateFlowDataLocal[start], size, MPI_DOUBLE, std::get<0>(DLB.transferNchem[DLB.recvFrom[i]]),
								  DLB.recvFrom[i] + 20000, MPI_COMM_WORLD, &sendRequests3[DLB.recvFrom[i]]);
				}
			}
		}
		t3 = MPI_Wtime();

		t4 = MPI_Wtime();
		tr2 += t4 - t3; /* trapezoid time */

		if (DLB.LoadDegree > DLB_tol)
		{
			int sentNchem = DLB.GetSentNchem(myid);
			DLB.recvUpdateFlowData.clear();
			DLB.recvUpdateFlowData.resize(DLB.sentNmesh * (18 + 4 * TwoDim.NS));
			/* Unpackage update data */
			if (sentNchem != 0)
			{
				double tt3 = MPI_Wtime();
				for (int i = 0; i < DLB.sendTo.size(); i++)
				{
					int start = std::accumulate(DLB.transferNmesh.begin() + DLB.sendTo[0], DLB.transferNmesh.begin() + DLB.sendTo[i], 0) * (18 + 4 * TwoDim.NS);
					int size = DLB.transferNmesh[DLB.sendTo[i]] * (18 + 4 * TwoDim.NS);
					if (size != 0)
					{
						// MPI_Recv(&DLB.recvUpdateFlowData[start], size, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
						// 		 DLB.sendTo[i] + 20000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						MPI_Irecv(&DLB.recvUpdateFlowData[start], size, MPI_DOUBLE, std::get<1>(DLB.transferNchem[DLB.sendTo[i]]),
								  DLB.sendTo[i] + 20000, MPI_COMM_WORLD, &recvRequests3[DLB.sendTo[i]]);
					}
				}
				double tt4 = MPI_Wtime();
				dt6 += tt4 - tt3;
			}

			// dt5 += tt2 - tt1;
			// dt6 += tt3 - tt2;
		}

		nchemSumBalance = 0;
		// #pragma omp parallel for num_threads(num_thread) collapse(3) schedule(dynamic, 100) reduction(+ : nchemSumBalance)
		for (int i = (ni - 1) / 2 + bc; i < ni + bc; i++)
			for (int j = bc; j < nj + bc; j++)
				for (int k = bc; k < nk + bc; k++)
				{
					int n = 0;
					bool flag = false;
					while (n < DLB.sentNmesh && tag < DLB.sentNmesh)
					{
						if (i == DLB.transferIndex[3 * n] &&
							j == DLB.transferIndex[3 * n + 1] &&
							k == DLB.transferIndex[3 * n + 2])
						{
							flag = true;
							tag++;
							break;
						}
						n++;
					}
					if (!flag)
					{
						nchemSumBalance += React.Nchem(i, j, k);
						for (int step = 0; step < React.Nchem(i, j, k); step++)
						{
							/* Solve the reaction source term */
							React.Trapezoid(TwoDim.Mc, TwoDim.Di, TwoDim.Yi, TwoDim.T, TwoDim.dt, step, i, j, k);

							/* Update the flow field */
							TwoDim.Explicit(step, i, j, k);
						}
					}
				}
		DLB.NchemSumBalance += nchemSumBalance;

		tw3 = MPI_Wtime();
		MPI_Waitall(DLB.transferNchem.size(), sendRequests3, statuses);
		MPI_Waitall(DLB.transferNchem.size(), recvRequests3, statuses);
		tw4 = MPI_Wtime();

		if (DLB.LoadDegree > DLB_tol)
		{
			int sentNchem = DLB.GetSentNchem(myid);
			if (sentNchem != 0)
				TwoDim.UnpackageUpdate(DLB.recvUpdateFlowData, DLB.transferIndex, DLB.sendTo, DLB.transferNmesh);
		}
		t5 = MPI_Wtime();

		// dt3 += t5 - t4;
		React.PushNchemMax();
		t6 = MPI_Wtime();
		// MPI_Barrier(MPI_COMM_WORLD);
		t7 = MPI_Wtime();
		tr3 += t3 - ttp2 + t5 - t4;	  /* DLB time */
		tr4 += tw4 - tw3 + tw2 - tw1; /* wait time */

		/* Monitor the reaction iteration step */
		if (iteration % DLB_step == 0)
		{
			MPI_Allgather(&DLB.NchemSumBalance, 1, MPI_INT, &React.NchemTotalBalance(0), 1, MPI_INT, MPI_COMM_WORLD);
			DLB.LoadDegreeBalance = double((React.NchemTotalBalance.MaxValue() - React.NchemTotalBalance.AveValue())) / React.NchemTotalBalance.MaxValue();
			if (myid == 0)
			{
				// cout << "After DLB: ";
				// React.NchemTotal.Print();
				cout << "Balance Load Degree: " << DLB.LoadDegreeBalance << endl;
				// cout << "Balance Load: ";
				// React.NchemTotalBalance.Print();
				// cout << "Diff: \n";
				// for (int i = 0; i < React.NchemTotal.GetSize(); i++)
				// 	cout << React.NchemTotal(i) - React.NchemTotalBalance(i) << " ";
				// cout << endl;
				// std::cout << "\tTransfer records:" << std::endl;
				// for (int i = 0; i < DLB.transferNchem.size(); i++)
				// {
				// 	int sender = std::get<0>(DLB.transferNchem[i]);
				// 	int receiver = std::get<1>(DLB.transferNchem[i]);
				// 	int amount = std::get<2>(DLB.transferNchem[i]);
				// 	std::cout << "\tSender: " << sender << " Receiver: " << receiver << " Transfer Nchem: " << amount << " Transfer Nmesh: " << DLB.transferNmesh[i] << std::endl;
				// }
			}
		}
		int max_it = 0;
		MPI_Reduce(&React.NchemNow, &max_it, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);
		if (myid == 0)
			cout << "MAX Reaction iteration step:  " << max_it << endl;
		// cout << "Process " << myid << ":  Reaction iteration step:  " << React.NchemNow << endl;
	}
}
/*---------------------------------------------------------------------------------
	Name:		Euler::IMEX

	Input(0):	None

	Function:	Solve the reaction term by IMEX method

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::IMEX()
{
	double t1 = 0, t2 = 0, t3 = 0;
	/* Solve the reaction source term */
	t1 = MPI_Wtime();
	React.Diagonalized(TwoDim.Mc, TwoDim.Di, TwoDim.T, TwoDim.Partial_T);
	t2 = MPI_Wtime();
	tr1 += t2 - t1;
	/* Update the flow field */
	TwoDim.Update_IMEX(React.CMS, React.MD);
	t3 = MPI_Wtime();
	tr2 += t3 - t2;
}

/*---------------------------------------------------------------------------------
	Name:		Euler::DNN

	Input(1):	The type of the reaction term solver

	Function:	Solve the reaction term by Cantera or DNN method
				(DNN (1) or Cantera (0))

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::DNN(int type)
{
	TwoDim.Update_after_Adv();

	// TwoDim.Di = React.ReactionS(type, TwoDim.Di, TwoDim.T, TwoDim.P, TwoDim.dt);

	TwoDim.Explicit();
}

/*---------------------------------------------------------------------------------
	Name:		NS::Output

	Input(1):	the number of the output file

	Function:	Output the result in a file

	Return:		None
 *---------------------------------------------------------------------------------*/

void Euler::Output(int num)
{
	int ni = Mymesh.ni;
	int nj = Mymesh.nj;
	int nk = Mymesh.nk;
	int bc = Mymesh.bc;

	/* Set the filename */
	string rea = react;
	string filename;
	if (numprocs == 1)
		filename = "./output_" + to_string(numprocs) + "/results/Series" + rea + to_string(num) + +"_" + to_string(myid) + ".dat";
	else
		filename = "./output_" + to_string(numprocs) + "/results/" + rea + to_string(num) + +"_" + to_string(myid) + ".dat";

	/* Output the result */
	ofstream outfile;
	outfile.open(filename, ios::out);
	/* Six significant digits hide small CPU/GPU differences and make a strict
	 * numerical comparison unreliable.  Seventeen digits round-trip a double. */
	outfile << std::setprecision(17);

	// /* The result without boundary */
	// outfile << "Variables = X,Y,Z,T,D,P,U,V,Ma,Y1,Y2,Y3,Y4,Y5,Y6,Y7,L,WaitTime\n";
	// outfile << "ZONE I=" << ni << '\t' << "J=" << nj << '\t' << "K=" << nk << '\n';
	// outfile << "datapacking=block\n";
	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.xnode(i) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.ynode(j) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << Mymesh.znode(k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.T(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.D(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.P(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.U(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.V(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Ma(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 0) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 1) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 2) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 3) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 4) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 5) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << TwoDim.Yi(i, j, k, 6) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << React.Nchem(i, j, k) << '\n';

	// for (int k = bc; k < nk + bc; k++)
	// 	for (int j = bc; j < nj + bc; j++)
	// 		for (int i = bc; i < ni + bc; i++)
	// 			outfile << tr4 << '\n';

	/* The result with boundary */
	outfile << "Variables = X,Y,Z,T,D,P,U,V,Ma,Y1,Y2,Y3,Y4,Y5,Y6,Y7\n";
	outfile << "ZONE I=" << ni + 2 * bc << '\t' << "J=" << nj + 2 * bc << '\t' << "K=" << nk + 2 * bc << '\n';
	outfile << "datapacking=block\n";
	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.xnode(i) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.ynode(j) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << Mymesh.znode(k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.T(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.D(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.P(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.U(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.V(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Ma(i, j, k) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 0) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 1) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 2) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 3) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 4) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 5) << '\n';

	for (int k = 0; k < nk + 2 * bc; k++)
		for (int j = 0; j < nj + 2 * bc; j++)
			for (int i = 0; i < ni + 2 * bc; i++)
				outfile << TwoDim.Yi(i, j, k, 6) << '\n';

	outfile.close();

	return;
}

/*---------------------------------------------------------------------------------
	Name:		NS::Output

	Input(1):	the number of the output file

	Function:	Output the result in a file

	Return:		None
 *---------------------------------------------------------------------------------*/
void Euler::Output_Total_Time(double t1, double t2, double t3, double t4, double t5, double t6, double t7, double t8, double t9, double t10)
{
	auto t12 = (t2 - t1);
	auto t23 = (t3 - t2);
	auto t34 = (t4 - t3);
	auto t45 = (t5 - t4);
	auto t56 = (t6 - t5);
	auto t67 = (t7 - t6);
	auto t78 = (t8 - t7);
	auto t89 = (t9 - t8);
	auto t90 = (t10 - t9);

	string filename = "./output_" + to_string(numprocs) + "/time_record/total/Total_Time_" + to_string(myid) + ".txt";
	ofstream outfile(filename);
	outfile << "myid:" << '\t' << myid << '\n';
	outfile << "FileRead:" << '\t' << t12 << '\n';
	outfile << "Mpi_Initial:" << '\t' << t23 << '\n';
	outfile << "MeshProcess:" << '\t' << t34 << '\n';
	outfile << "InputRead:" << '\t' << t45 << '\n';
	outfile << "ReactionRead:" << '\t' << t56 << '\n';
	outfile << "FieldInitial:" << '\t' << t67 << '\n';
	outfile << "MPI_Barrier:" << '\t' << t78 << '\n';
	outfile << "Computing:" << '\t' << t89 << '\n';
	outfile << "Output:" << '\t' << t90 << '\n';
	outfile.close();

	double maxt = 0;
	MPI_Reduce(&t89, &maxt, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (myid == 0)
	{
		string rea = react;
		filename = "./output_" + to_string(numprocs) + "/time_record/compare/" + rea + "_mpi_" + to_string(m_block_x * m_block_y) + "_omp_" + to_string(num_thread) + ".txt";
		ofstream outfile1(filename);
		outfile1 << maxt << '\t' << trackt;
		outfile1.close();
		std::cout << "Loop time: " << maxt << std::endl;
	}
}
void Euler::Output_Compute_Time(double t12, double t23, double t34, double t45, double t67)
{
	string file_compute = "./output_" + to_string(numprocs) + "/time_record/compute/Compute_Time_" + to_string(myid) + ".txt";
	ofstream outfile(file_compute);
	outfile << "myid:" << '\t' << myid << '\n';
	outfile << "FieldBoundary:" << '\t' << t12 << '\n';
	outfile << "Mpi_Boundary:" << '\t' << t23 << '\n';
	outfile << "Advection:" << '\t' << t34 << '\n';
	outfile << "Reaction:" << '\t' << t45 << '\n';
	outfile << "Output:" << '\t' << t67 << '\n';
	switch (Reaction_Sch)
	{
	case 0:
		outfile << "Update_after_Advection:" << '\t' << tr1 << '\n';
		outfile << "Trapezoid_Prediction:" << '\t' << dt4 << '\n';
		outfile << "Trapezoid:" << '\t' << tr2 + dt3 << '\n';
		outfile << "DLB:" << '\t' << tr3 - dt3 << '\n';
		outfile << "Wait:" << '\t' << tr4 << '\n';
		break;
	case 1:
		outfile << "Diagnalized:" << '\t' << tr1 << '\n';
		outfile << "Update_IMEX:" << '\t' << tr2 << '\n';
		outfile << "FieldBoundary:" << TwoDim.ft1 << " " << TwoDim.ft2 << " " << TwoDim.ft3 << " " << TwoDim.ft4 << " " << TwoDim.ft5 << " " << TwoDim.ft6 << '\n';
		break;
	}

	outfile.close();
	double recv_wait_time[m_block_x * m_block_y];
	double recv_react_time[m_block_x * m_block_y];
	double recv_advect_time[m_block_x * m_block_y];
	double recv_DLB_time[m_block_x * m_block_y];
	double recv_DLB_time1[m_block_x * m_block_y];
	double recv_DLB_time2[m_block_x * m_block_y];
	double recv_DLB_time3[m_block_x * m_block_y];
	double recv_DLB_time4[m_block_x * m_block_y];
	double recv_DLB_time5[m_block_x * m_block_y];
	double recv_DLB_time6[m_block_x * m_block_y];
	double react_time = tr1 + tr2 + dt3 + dt4;
	double dlb_time = tr3 - dt3;
	MPI_Gather(&tr4, 1, MPI_DOUBLE, &recv_wait_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&react_time, 1, MPI_DOUBLE, &recv_react_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&t34, 1, MPI_DOUBLE, &recv_advect_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dlb_time, 1, MPI_DOUBLE, &recv_DLB_time, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt1, 1, MPI_DOUBLE, &recv_DLB_time1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt2, 1, MPI_DOUBLE, &recv_DLB_time2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt3, 1, MPI_DOUBLE, &recv_DLB_time3, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt5, 1, MPI_DOUBLE, &recv_DLB_time4, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt6, 1, MPI_DOUBLE, &recv_DLB_time5, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Gather(&dt7, 1, MPI_DOUBLE, &recv_DLB_time6, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (myid == 0)
	{
		string file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_wait.txt";
		ofstream outfile1(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile1 << recv_wait_time[i] << endl;
		outfile1.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_react.txt";
		ofstream outfile2(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile2 << recv_react_time[i] << endl;
		outfile2.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_advect.txt";
		ofstream outfile3(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile3 << recv_advect_time[i] << endl;
		outfile3.close();

		file_total_time = "./output_" + to_string(numprocs) + "/time_record/all/All_DLB.txt";
		ofstream outfile4(file_total_time);
		for (int i = 0; i < m_block_x * m_block_y; i++)
			outfile4 << recv_DLB_time[i] << '\t' << recv_DLB_time1[i] << '\t' << recv_DLB_time2[i] << '\t'
					 << recv_DLB_time3[i] << '\t' << recv_DLB_time4[i] << '\t' << recv_DLB_time5[i] << '\t' << recv_DLB_time6[i] << endl;
		outfile4.close();
	}
}

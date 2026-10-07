/*=================================================================================
 * The functions of Reaction
 *=================================================================================*/

/*---------------------------------------------------------------------------------
 * Standard C++ library headers
 *---------------------------------------------------------------------------------*/

#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>
#include <mpi.h>
#include <cmath>
#include <numeric>

/*---------------------------------------------------------------------------------
 * Local headers
 *---------------------------------------------------------------------------------*/

#include "Reaction.hpp"
#include "FileReader.hpp"
#include "Function.hpp"
#include "Global.hpp"
// #include "cantera/thermo.hpp"
// #include "cantera/kinetics.hpp"
// #include "cantera/base/Solution.hpp"
// #include "cantera/zeroD/IdealGasReactor.hpp"
// #include "cantera/zeroD/ReactorNet.hpp"
// #include "cantera/base/Array.hpp"
// #include "cantera/zerodim.hpp"
// #include "cantera/thermo/IdealGasPhase.hpp"
// #include "cantera/numerics/Integrator.hpp"

using namespace std;

/*---------------------------------------------------------------------------------
    Name:		Reaction::ReactionConstruction

    Input(0):	None

    Function:	Constuct arrays to store information of the reaction

    Return:		None
 *---------------------------------------------------------------------------------*/

void Reaction::ReactionConstruction()
{
    Stoi_F.Initial(NS, NR);
    Stoi_B.Initial(NS, NR);

    Af.Initial(NR);
    Bf.Initial(NR);
    Eaf.Initial(NR);

    React_TB.Initial(NS, NR);

    Mr_temp.Initial(NS);
    Mr.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);
    Mi.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);
    Yi.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);

    Coeff0.Initial(9, NS);
    Coeff1.Initial(9, NS);
    Mw.Initial(NS);
    Ri.Initial(NS);

    Hi.Initial(NS);
    Si.Initial(NS);
    Gi.Initial(NS);

    KF.Initial(NR);
    KB.Initial(NR);
    Kp.Initial(NR);
    Kc.Initial(NR);

    RR_F.Initial(NR);
    RR_B.Initial(NR);
    R_TB.Initial(NR);
    RR.Initial(NR);
    Mc.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);
    Di.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);
    Wi.Initial(NS);
    CMS.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS);

    WJH1.Initial(NS, NR);
    WJH2.Initial(NS, NR);
    MD.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc, NS + 3, NS + 3);

    P.Initial(NS);
    Q.Initial(NS);

    Nchem.Initial(ni + 2 * bc, nj + 2 * bc, nk + 2 * bc);
    Nchem.Fill(1);
    NchemTotal.Initial(numprocs);
    NchemTotalBalance.Initial(numprocs);
}
void Reaction::ReConstruction(int meshnum)
{
    Mr.Initial(meshnum, 1, 1, NS);
    Mi.Initial(meshnum, 1, 1, NS);
    Yi.Initial(meshnum, 1, 1, NS);
    Mc.Initial(meshnum, 1, 1, NS);
    Di.Initial(meshnum, 1, 1, NS);
    CMS.Initial(meshnum, 1, 1, NS);
    MD.Initial(meshnum, 1, 1, NS + 3, NS + 3);
    Nchem.Initial(meshnum, 1, 1);
    Nchem.Fill(1);
}
/*---------------------------------------------------------------------------------
    Name:		ifstream& seek_to_line

    Input(2):	The file; The selected line

    Function:	Locate to the selected line

    Return:		None
 *---------------------------------------------------------------------------------*/

ifstream &seek_to_line(ifstream &in, int line)
{
    int i;
    char buf[1024];

    /* Locate to the beginning line of the file */
    in.seekg(0, ios::beg);

    /* Locate to the selected line */
    for (i = 0; i < line; i++)
    {
        in.getline(buf, sizeof(buf));
    }

    return in;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::ReactionRead

    Input(4):	The mesh nodes; the reaction model; the thermo file

    Function:	Read the reaction model and thermo file;
                Construct arrays to store the information

    Return:		None
 *---------------------------------------------------------------------------------*/

void Reaction::ReactionRead(char *reaction_model, char *thermofile, Array<double, 1> &xnode, Array<double, 1> &ynode, Array<double, 1> &znode, int bc)
{
    this->xnode = xnode;
    this->ynode = ynode;
    this->znode = znode;
    this->bc = bc;

    ni = xnode.GetSize() - 2 * bc;
    nj = ynode.GetSize() - 2 * bc;
    nk = znode.GetSize() - 2 * bc;

    ifstream fin;
    fin.open(reaction_model);
    int a[100] = {0}, i = 0, line = 2;
    string s, item;
    seek_to_line(fin, line);
    getline(fin, s, '\n');

    /* Transfer string to number */
    stringstream text_stream(s);

    /* Read the number of species, reactions and three-body reactions */
    while (std::getline(text_stream, item, '\t'))
    {
        a[i] = stoi(item);
        i++;
    }
    NS = a[0];
    NR = a[1];
    TB = a[2];
    // cout << NS << '\t' << NR << '\t' << TB << endl;
    // cout << endl;

    /* Construction */
    ReactionConstruction();

    /* Read the stoichiometric coefficient */
    line = line + 6;
    seek_to_line(fin, line);

    for (int j = 0; j < NR; j++)
    {
        i = 0;
        getline(fin, s, '\n');
        stringstream reaction_forward(s);
        while (std::getline(reaction_forward, item, '\t'))
        {
            Stoi_F(i, j) = stoi(item);
            // cout << Stoi_F(i, j) << '\t';
            i++;
        }
        // cout << endl;
        i = 0;
        getline(fin, s, '\n');
        stringstream reaction_backward(s);
        while (std::getline(reaction_backward, item, '\t'))
        {
            Stoi_B(i, j) = stoi(item);
            // cout << Stoi_B(i, j) << '\t';
            i++;
        }
        // cout << endl;
    }
    // cout << endl;

    /* Read the pre-exponent factor, temperature index and activation energy */
    line = line + 2 * NR + 2;
    seek_to_line(fin, line);

    for (int j = 0; j < NR; j++)
    {
        i = 0;
        double b[4] = {0};
        while (fin >> s)
        {
            stringstream geek(s);
            geek >> b[i];
            i++;
            if (i == 4)
                break;
        }
        Af(j) = b[1];
        Bf(j) = b[2];
        Eaf(j) = b[3];
        // cout  << Af(j) << '\t' << Bf(j) << '\t' << Eaf(j) << endl;
        // cout << endl;
    }
    // cout << endl;

    /* Read the coefficient of three-body reactions */
    line = line + NR + 3;
    seek_to_line(fin, line);
    React_TB.Fill(0.0);
    for (int j = 0; j < TB; j++)
    {
        i = 0;
        while (fin >> s)
        {
            stringstream geek(s);
            geek >> a[i];
            i++;
            if (i == NS + 1)
                break;
        }

        for (int k = 0; k < NS; k++)
        {
            React_TB(k, a[0] - 1) = a[k + 1];
            // cout << React_TB(k, a[0] - 1) << '\t';
        }
        // cout << endl;
    }
    // cout << endl;

    /* Read the component mole ratio */
    line = line + TB + 2;
    seek_to_line(fin, line);
    i = 0;
    double c[100] = {0};
    while (fin >> s)
    {
        stringstream geek(s);
        geek >> c[i];
        i++;
        if (i == NS)
            break;
    }
    for (int i = 0; i < NS; i++)
    {
        Mr_temp(i) = c[i];
        cout << Mr_temp(i) << '\t';
    }
    cout << endl;
    fin.close();

    ifstream in;
    in.open(thermofile);

    /* Read the NASA Thermochemical polynomial coefficient and component molecular weight */
    line = 15;
    double aa[100] = {0};

    for (int j = 0; j < NS; j++)
    {
        seek_to_line(in, line);
        i = 0;
        while (in >> s)
        {
            stringstream geek(s);
            geek >> aa[i];
            // cout << aa[i] << endl;
            i++;
            if (i == 19)
                break;
        }
        for (int k = 0; k < 9; k++)
        {
            Coeff0(k, j) = aa[k];
            // cout << setiosflags(ios::scientific) << setprecision(9) <<Coeff0(k, j) << '\t';
        }
        // cout << endl;
        for (int k = 0; k < 9; k++)
        {
            Coeff1(k, j) = aa[k + 9];
            // cout << setiosflags(ios::scientific) << setprecision(9)<< Coeff1(k, j) << '\t';
        }
        // cout << endl;
        Mw(j) = aa[18];
        // cout << setiosflags(ios::scientific) << setprecision(7) << Mw(j) << endl;
        line = line + 6;
    }
    fin.close();

    ReactionInitial();

    return;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::ReactionInitial

    Input(0):	None

    Function:	Calculate the component gas constant, the component mole fraction
                and the mass fraction

    Return:		None
 *---------------------------------------------------------------------------------*/

void Reaction::ReactionInitial()
{
    Function Fun;
    Yi.Fill(0.0);
    Mi.Fill(0.0);
    Mr.Fill(0.0);

    /* RSBI */
    /* Bubble condition */
    double Bubble_CenterX = 0.038;      /* x coordinate */
    double Bubble_CenterY = 0.0;        /* y coordinate */
    double Bubble_Ir = 0.022;           /* Inner radius */
    double Bubble_Or = 1.2 * Bubble_Ir; /* Outer radius */
    double alpha = 5;
    for (int s = 0; s < NS; s++)
        Ri(s) = R / Mw(s);
    for (int i = 0; i < ni + 2 * bc; i++)
        for (int j = 0; j < nj + 2 * bc; j++)
            for (int k = 0; k < nk + 2 * bc; k++)
            {
                double Mr_SUM = 0.0;
                double Total_Mw = 0.0;
                if (pow(xnode(i) - Bubble_CenterX, 2) + pow(ynode(j) - Bubble_CenterY, 2) > pow(Bubble_Or, 2))
                {
                    Mr(i, j, k, 4) = 1.0;
                    Mr(i, j, k, 8) = 3.76;
                    for (int s = 0; s < NS; s++)
                    {
                        Mr_SUM += Mr(i, j, k, s);
                        Total_Mw += Mr(i, j, k, s) * Mw(s);
                    }
                    for (int s = 0; s < NS; s++)
                    {
                        Mi(i, j, k, s) = Mr(i, j, k, s) / Mr_SUM;
                        Yi(i, j, k, s) = Mr(i, j, k, s) * Mw(s) / Total_Mw;
                    }
                }
                else if (pow(xnode(i) - Bubble_CenterX, 2) + pow(ynode(j) - Bubble_CenterY, 2) < pow(Bubble_Ir, 2))
                {
                    Mr(i, j, k, 5) = 1.0;
                    for (int s = 0; s < NS; s++)
                    {
                        Mr_SUM += Mr(i, j, k, s);
                        Total_Mw += Mr(i, j, k, s) * Mw(s);
                    }
                    for (int s = 0; s < NS; s++)
                    {
                        Mi(i, j, k, s) = Mr(i, j, k, s) / Mr_SUM;
                        Yi(i, j, k, s) = Mr(i, j, k, s) * Mw(s) / Total_Mw;
                    }
                }
                else
                {
                    double r = sqrt(pow(xnode(i) - Bubble_CenterX, 2) + pow(ynode(j) - Bubble_CenterY, 2));
                    Yi(i, j, k, 5) = exp(-alpha * pow((r - Bubble_Ir) / (Bubble_Or - Bubble_Ir), 2));
                    Yi(i, j, k, 4) = (1 - Yi(i, j, k, 5)) * Mw(4) / (Mw(4) + 3.76 * Mw(8));
                    Yi(i, j, k, 8) = (1 - Yi(i, j, k, 5)) * 3.76 * Mw(8) / (Mw(4) + 3.76 * Mw(8));
                }
            }
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::Trapezoid

    Input(4):	The component mole concentration; The density fraction;
                The temperature; The time step

    Function:	Simulate the chemical reaction process by Trapezoid formula
                Calculate the component mass production rate and density fraction

    Return:		The density fraction
 *---------------------------------------------------------------------------------*/

Array<double, 4> Reaction::Trapezoid(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 3> &T, double dt)
{
    // //#pragma omp parallel for collapse(2) num_threads(num_thread) default(shared), private(ni, nj, NS, bc)

    for (int i = bc; i < ni + bc; i++)
        for (int j = bc; j < nj + bc; j++)
            for (int k = bc; k < nk + bc; k++)
            {
                T_local = T(i, j, k);
                for (int s = 0; s < NS; s++)
                {
                    Di(i, j, k, s) = Di_temp(i, j, k, s) * 1e-3;
                    Mc(i, j, k, s) = Mc_temp(i, j, k, s) * 1e-6;
                    Hi(s) = GetHi(T_local, R, s, Coeff0, Coeff1);
                    Si(s) = GetSi(T_local, R, s, Coeff0, Coeff1);
                    Gi(s) = Hi(s) - Si(s) * T_local;
                }

                /* Calculate the net reaction rate */
                RR_F.Fill(1.0);
                RR_B.Fill(1.0);
                R_TB.Fill(0);
                for (int r = 0; r < NR; r++)
                {
                    double Xr = 0.0, delta_g = 0.0;
                    KF(r) = Af(r) * pow(T_local, Bf(r)) * exp(-Eaf(r) / (Ru * T_local));
                    for (int s = 0; s < NS; s++)
                    {
                        Xr += (Stoi_B(s, r) - Stoi_F(s, r));
                        delta_g += (Stoi_B(s, r) - Stoi_F(s, r)) * Gi(s);
                        RR_F(r) *= pow(Mc(i, j, k, s), Stoi_F(s, r));
                        RR_B(r) *= pow(Mc(i, j, k, s), Stoi_B(s, r));
                        R_TB(r) += Mc(i, j, k, s) * React_TB(s, r);
                    }
                    Kp(r) = exp(-delta_g / (R * T_local)) * pow((P0 * 1e-6), Xr);
                    Kc(r) = Kp(r) * pow(R * T_local, -Xr);
                    KB(r) = KF(r) / Kc(r);
                    if (R_TB(r) == 0.0)
                        R_TB(r) = 1.0;
                    RR(r) = KF(r) * RR_F(r) - KB(r) * RR_B(r);
                }

                /* Calculate the mass prodcution rate */
                for (int s = 0; s < NS; s++)
                {
                    double temp = 0.0;
                    for (int r = 0; r < NR; r++)
                        temp += (Stoi_B(s, r) - Stoi_F(s, r)) * R_TB(r) * RR(r);
                    Wi(s) = Mw(s) * temp;
                    CMS(i, j, k, s) = Wi(s) * 1e3;
                }

                P.Fill(0), Q.Fill(0);
                for (int s = 0; s < NS; s++)
                {
                    for (int r = 0; r < NR; r++)
                    {
                        Q(s) += (Stoi_B(s, r) * KF(r) * RR_F(r) + Stoi_F(s, r) * KB(r) * RR_B(r)) * R_TB(r);
                        P(s) += (Stoi_B(s, r) * KB(r) * RR_B(r) + Stoi_F(s, r) * KF(r) * RR_F(r)) * R_TB(r);
                    }
                    Q(s) = Mw(s) * Q(s);
                    if (Mc(i, j, k, s) == 0)
                    {
                        P(s) = 0.0;
                    }
                    else
                    {
                        P(s) = P(s) / Mc(i, j, k, s);
                    }
                    Di(i, j, k, s) = ((1.0 - dt / 2.0 * P(s)) * Di(i, j, k, s) + dt * Q(s)) / (1.0 + dt / 2.0 * P(s)) * 1e3;
                }
            }

    return Di;
}

/* Adaptive */
void Reaction::Trapezoid(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 4> &Yi_temp, Array<double, 3> &T, double dtm, int step, int i, int j, int k)
{
    double dt = 0.0, dt_temp = 0.0;
    double T_local;
    intermidatePara __attribute__((aligned(64))) ip;

    // dt = dtm;
    // if (step > 0)
    dt = dtm / Nchem(i, j, k);

    T_local = T(i, j, k);
    for (int s = 0; s < NS; s++)
    {
        Di(i, j, k, s) = Di_temp(i, j, k, s) * 1e-3;
        Mc(i, j, k, s) = Mc_temp(i, j, k, s) * 1e-6;
        ip.Hi[s] = GetHi(T_local, R, s, Coeff0, Coeff1);
        ip.Si[s] = GetSi(T_local, R, s, Coeff0, Coeff1);
        ip.Gi[s] = ip.Hi[s] - ip.Si[s] * T_local;
    }

    /* Calculate the net reaction rate */
    std::fill(ip.RR_F, ip.RR_F + NR, 1.0);
    std::fill(ip.RR_B, ip.RR_B + NR, 1.0);
    std::fill(ip.R_TB, ip.R_TB + NR, 0.0);
    for (int r = 0; r < NR; r++)
    {
        double Xr = 0.0, delta_g = 0.0;
        ip.KF[r] = Af(r) * pow(T_local, Bf(r)) * exp(-Eaf(r) / (Ru * T_local));
        for (int s = 0; s < NS; s++)
        {
            Xr += (Stoi_B(s, r) - Stoi_F(s, r));

            delta_g += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.Gi[s];
            ip.RR_F[r] *= pow(Mc(i, j, k, s), Stoi_F(s, r));
            ip.RR_B[r] *= pow(Mc(i, j, k, s), Stoi_B(s, r));
            ip.R_TB[r] += Mc(i, j, k, s) * React_TB(s, r);
            // Disable per-substep debug output.
            // if (i == 100 && j == 70 && k == 2)
            //     cout << "Stoi_B(s, r) " << Stoi_B(s, r) << " Stoi_F(s, r) " << Stoi_F(s, r) << " Gi " << ip.Gi[s] << " delta_g " << delta_g << endl;
        }
        ip.Kp[r] = exp(-delta_g / (R * T_local)) * pow((P0 * 1e-6), Xr);
        ip.Kc[r] = ip.Kp[r] * pow(R * T_local, -Xr);
        ip.KB[r] = ip.KF[r] / ip.Kc[r];
        if (ip.R_TB[r] == 0.0)
            ip.R_TB[r] = 1.0;
        ip.RR[r] = ip.KF[r] * ip.RR_F[r] - ip.KB[r] * ip.RR_B[r];
    }

    /* Calculate the mass prodcution rate */
    for (int s = 0; s < NS; s++)
    {
        double temp = 0.0;
        for (int r = 0; r < NR; r++)
            temp += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.R_TB[r] * ip.RR[r];
        ip.Wi[s] = Mw(s) * temp;
        CMS(i, j, k, s) = ip.Wi[s] * 1e3;
    }

    /* Calculate dt for each mesh node */
    // if (step == 0)
    // {
    //     dt = dtm;
    //     for (int s = 0; s < NS; s++)
    //     {
    //         if (Yi_temp(i, j, k, s) >= 1e-6)
    //         {
    //             if (ip.Wi[s] != 0)
    //             {
    //                 dt_temp = abs(-Di(i, j, k, s) / ip.Wi[s]);
    //                 if (dt_temp < dt)
    //                     dt = dt_temp;
    //             }
    //         }
    //     }
    //     Nchem(i, j, k) = int(ceil(dtm / dt));
    //     // Nchem(i, j, k) = 1;
    //     dt = dtm / Nchem(i, j, k);
    //     if (Nchem(i, j, k) <= 0)
    //     {
    //         cout << "wrong chem " << '\t' << dtm << '\t' << dt << endl;
    //         abort();
    //     }
    // }

    std::fill(ip.P, ip.P + NS, 0.0);
    std::fill(ip.Q, ip.Q + NS, 0.0);
    for (int s = 0; s < NS; s++)
    {
        for (int r = 0; r < NR; r++)
        {
            ip.Q[s] += (Stoi_B(s, r) * ip.KF[r] * ip.RR_F[r] + Stoi_F(s, r) * ip.KB[r] * ip.RR_B[r]) * ip.R_TB[r];
            ip.P[s] += (Stoi_B(s, r) * ip.KB[r] * ip.RR_B[r] + Stoi_F(s, r) * ip.KF[r] * ip.RR_F[r]) * ip.R_TB[r];
        }
        ip.Q[s] = Mw(s) * ip.Q[s];
        if (Mc(i, j, k, s) == 0)
        {
            ip.P[s] = 0.0;
        }
        else
        {
            ip.P[s] = ip.P[s] / Mc(i, j, k, s);
        }
        Di_temp(i, j, k, s) = ((1.0 - dt / 2.0 * ip.P[s]) * Di(i, j, k, s) + dt * ip.Q[s]) / (1.0 + dt / 2.0 * ip.P[s]) * 1e3;
    }

    // return Di;
}

void Reaction::TrapezoidPrediction(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 4> &Yi_temp, Array<double, 3> &T, double dtm)
{
    // #pragma omp parallel for num_threads(num_thread) collapse(3)
    for (int i = bc; i < ni + bc; i++)
        for (int j = bc; j < nj + bc; j++)
            for (int k = bc; k < nk + bc; k++)
            {
                double dt = dtm, dt_temp = 0.0;
                double T_local = T(i, j, k);
                intermidatePara __attribute__((aligned(64))) ip;

                for (int s = 0; s < NS; s++)
                {
                    Di(i, j, k, s) = Di_temp(i, j, k, s) * 1e-3;
                    Mc(i, j, k, s) = Mc_temp(i, j, k, s) * 1e-6;
                    ip.Hi[s] = GetHi(T_local, R, s, Coeff0, Coeff1);
                    ip.Si[s] = GetSi(T_local, R, s, Coeff0, Coeff1);
                    ip.Gi[s] = ip.Hi[s] - ip.Si[s] * T_local;
                }

                /* Calculate the net reaction rate */
                std::fill(ip.RR_F, ip.RR_F + NR, 1.0);
                std::fill(ip.RR_B, ip.RR_B + NR, 1.0);
                std::fill(ip.R_TB, ip.R_TB + NR, 0.0);
                for (int r = 0; r < NR; r++)
                {
                    double Xr = 0.0, delta_g = 0.0;
                    ip.KF[r] = Af(r) * pow(T_local, Bf(r)) * exp(-Eaf(r) / (Ru * T_local));
                    for (int s = 0; s < NS; s++)
                    {
                        Xr += (Stoi_B(s, r) - Stoi_F(s, r));
                        delta_g += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.Gi[s];
                        ip.RR_F[r] *= pow(Mc(i, j, k, s), Stoi_F(s, r));
                        ip.RR_B[r] *= pow(Mc(i, j, k, s), Stoi_B(s, r));
                        ip.R_TB[r] += Mc(i, j, k, s) * React_TB(s, r);
                        // cout << React_TB(s, r) << " ";
                    }
                    // cout << endl;
                    ip.Kp[r] = exp(-delta_g / (R * T_local)) * pow((P0 * 1e-6), Xr);
                    ip.Kc[r] = ip.Kp[r] * pow(R * T_local, -Xr);
                    ip.KB[r] = ip.KF[r] / ip.Kc[r];
                    if (ip.R_TB[r] == 0.0)
                        ip.R_TB[r] = 1.0;
                    ip.RR[r] = ip.KF[r] * ip.RR_F[r] - ip.KB[r] * ip.RR_B[r];
                }

                /* Calculate the mass prodcution rate */
                for (int s = 0; s < NS; s++)
                {
                    double temp = 0.0;
                    for (int r = 0; r < NR; r++)
                        temp += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.R_TB[r] * ip.RR[r];
                    ip.Wi[s] = Mw(s) * temp;
                    // CMS(i, j, k, s) = ip.Wi[s] * 1e3;
                }

                /* Calculate dt for each mesh node */
                for (int s = 0; s < NS; s++)
                {
                    if (Yi_temp(i, j, k, s) >= 1e-6)
                    {
                        if (ip.Wi[s] != 0)
                        {
                            dt_temp = abs(-Di(i, j, k, s) / ip.Wi[s]);
                            if (dt_temp < dt)
                                dt = dt_temp;
                        }
                    }
                }
                Nchem(i, j, k) = int(ceil(dtm / dt));

                if (Nchem(i, j, k) <= 0)
                {
                    cout << "wrong chem " << '\t' << dtm << '\t' << dt << endl;
                    abort();
                }
            }
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::PushNchemMax

    Input(0):	None

    Function:	Push the maximum Nchem for whole computation domain

    Return:		None
 *---------------------------------------------------------------------------------*/
void Reaction::PushNchemMax()
{
    NchemNow = Nchem.MaxValue();
    NchemMax_Rank.push_back(NchemNow);
}
/*---------------------------------------------------------------------------------
    Name:		Reaction::GetNchemMax

    Input(0):	None

    Function:	Get the maximum Nchem for whole computation domain

    Return:		None
 *---------------------------------------------------------------------------------*/
void Reaction::GetNchemMax()
{
    int size = NchemMax_Rank.size();
    NchemMax_Total.Initial(size * numprocs);
    MPI_Gather(&NchemMax_Rank[0], size, MPI_INT, &NchemMax_Total(0), size, MPI_INT, 0, MPI_COMM_WORLD);
    Array<int, 1> NchemMax_temp;
    NchemMax_temp.Initial(numprocs);
    if (myid == 0)
    {
        for (int j = 0; j < size; j++)
        {
            for (int i = 0; i < numprocs; i++)
            {
                NchemMax_temp(i) = NchemMax_Total(j + i * size);
            }
            NchemMax.push_back(NchemMax_temp.MaxValue());
        }
        ofstream outfile("./output_" + to_string(numprocs) + "/load/Load_Loop.dat");
        for (int i = 1; i <= NchemMax.size(); i++)
            outfile << i << '\t' << NchemMax[i - 1] << '\n';
    }
}

/* Restore transfer mesh data */
// void Reaction::Package(vector<double> &data, int i, int j, int k)
// {

// }

/*---------------------------------------------------------------------------------
    Name:		Reaction::Diagonalized

    Input(4):	The component mole concentration; The density fraction;
                The temperature; The partial T

    Function:	Simulate the chemical reaction process by IMEX method
                Calculate the component mass production rate and diagonal of the Jacobi matrix

    Return:		None
 *---------------------------------------------------------------------------------*/

void Reaction::Diagonalized(Array<double, 4> &Mc_temp, Array<double, 4> &Di_temp, Array<double, 3> &T, Array<double, 4> &Partial_T)
{
    // #pragma omp parallel for num_threads(num_thread) collapse(3) schedule(static)
    for (int i = bc; i < ni + bc; i++)
        for (int j = bc; j < nj + bc; j++)
            for (int k = bc; k < nk + bc; k++)
            {
                interParaDiag __attribute__((aligned(64))) ip;
                const double T_local = T(i, j, k);
                for (int s = 0; s < NS; s++)
                {
                    Di(i, j, k, s) = Di_temp(i, j, k, s) * 1e-3;
                    Mc(i, j, k, s) = Mc_temp(i, j, k, s) * 1e-6;
                    ip.Hi[s] = GetHi(T_local, R, s, Coeff0, Coeff1);
                    ip.Si[s] = GetSi(T_local, R, s, Coeff0, Coeff1);
                    ip.Gi[s] = ip.Hi[s] - ip.Si[s] * T_local;
                }

                /* Calculate the net reaction rate */
                std::fill(ip.RR_F, ip.RR_F + NR, 1.0);
                std::fill(ip.RR_B, ip.RR_B + NR, 1.0);
                std::fill(ip.R_TB, ip.R_TB + NR, 0.0);
                for (int r = 0; r < NR; r++)
                {
                    double Xr = 0.0, delta_g = 0.0;
                    ip.KF[r] = Af(r) * pow(T_local, Bf(r)) * exp(-Eaf(r) / (Ru * T_local));
                    for (int s = 0; s < NS; s++)
                    {
                        Xr += (Stoi_B(s, r) - Stoi_F(s, r));
                        delta_g += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.Gi[s];
                        ip.RR_F[r] *= pow(Mc(i, j, k, s), Stoi_F(s, r));
                        ip.RR_B[r] *= pow(Mc(i, j, k, s), Stoi_B(s, r));
                        ip.R_TB[r] += Mc(i, j, k, s) * React_TB(s, r);
                    }
                    ip.Kp[r] = exp(-delta_g / (R * T_local)) * pow((P0 * 1e-6), Xr);
                    ip.Kc[r] = ip.Kp[r] * pow(R * T_local, -Xr);
                    ip.KB[r] = ip.KF[r] / ip.Kc[r];
                    if (ip.R_TB[r] == 0.0)
                        ip.R_TB[r] = 1.0;
                    ip.RR[r] = ip.KF[r] * ip.RR_F[r] - ip.KB[r] * ip.RR_B[r];
                }

                /* Calculate the mass prodcution rate */
                for (int s = 0; s < NS; s++)
                {
                    double temp = 0.0;
                    for (int r = 0; r < NR; r++)
                        temp += (Stoi_B(s, r) - Stoi_F(s, r)) * ip.R_TB[r] * ip.RR[r];
                    ip.Wi[s] = Mw(s) * temp;
                    CMS(i, j, k, s) = ip.Wi[s] * 1e3;
                }

                /* Calculate the diagonal of the Jacobi matrix */
                for (int r = 0; r < NR; r++)
                    for (int s = 0; s < NS; s++)
                    {
                        if (Di(i, j, k, s) == 0)
                        {
                            ip.WJH1[s][r] = 0.0;
                        }
                        else
                        {
                            ip.WJH1[s][r] = ip.KF[r] * Stoi_F(s, r) * ip.RR_F[r] / Di(i, j, k, s) - ip.KB[r] * Stoi_B(s, r) * ip.RR_B[r] / Di(i, j, k, s);
                        }
                        ip.WJH2[s][r] = ip.RR[r] * (Bf[r] / T_local + Eaf(r) / (Ru * pow(T_local, 2)));
                    }

                for (int s = 0; s < NS; s++)
                {
                    double temp = 0.0;
                    for (int r = 0; r < NR; r++)
                        temp += (Stoi_B(s, r) - Stoi_F(s, r)) * ((ip.WJH1[s][r] + ip.WJH2[s][r] * Partial_T(i, j, k, s)) * ip.R_TB[r] + ip.RR[r] * React_TB(s, r) / Mw(s));
                    MD(i, j, k, s, s) = Mw(s) * temp;
                }
            }
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::ReactionS

    Input(4):	The solver type; The density fraction;
                The temperature; The pressure

    Function:	Simulate the chemical reaction process
                Calculate the component mass production rate
                difference is that the solver is Cantera or DNN

    Return:		The updated component density fraction
 *---------------------------------------------------------------------------------*/

// Array<double> Reaction::ReactionS(int type, Array<double> Di_temp, Array<double> T, Array<double> P, double dt)
// {
//    double frac_t;
//    double tot_t = dt;
//    double sub_t = 0.0;
//    double T_local, P_local, D_local;
//    int pair[9] = { 5, 4, 2, 1, 3, 0, 7, 6, 8 }, v = 0;
//    D_local = 0.0;
//    double* Di = new double[NS];
//    if (type == 1)
//    {
//        auto sol = Cantera::newSolution("JMH2.yaml", "gas", "none");
//        auto gas = sol->thermo();

//        Cantera::IdealGasReactor r;
//        for (int j = 2; j < nj + 2; j++)
//        {
//            for (int i = 2; i < ni + 2; i++)
//            {
//                for (v = 0; v < NS; v++)
//                {
//                    if (pair[v] >= 0) Di[v] = Di_temp(i, j, pair[v]);
//                    else Di[v] = 0.0;
//                    D_local += Di[v];
//                }
//                sub_t = 0.0;
//                T_local = T(i, j);
//                P_local = P(i, j);
//                //different frac_t for different zones
//                if (T_local < 500) frac_t = 5e-8;
//                else frac_t = 1e-8;
//                for (int k = 0; k < NS; k++) Di[k] = Di[k] / D_local;
//                gas->setState_TPY(T_local, P_local, Di);
//                r.insert(sol);
//                Cantera::ReactorNet net;
//                net.addReactor(r);
//                net.setMaxTimeStep(frac_t);
//                net.advance(dt);

//                for (int s = 0; s < NS; s++)
//                {
//                    for (v = 0; v < NS; v++) if (pair[v] == s) Di_temp(i, j, s) = r.massFraction(v) * r.density();
//                }

//            }
//        }

//    }
//    else if (type == 2) //DNN
//    {

//    }
//    delete[] Di;
//    return Di_temp;

// }

/*---------------------------------------------------------------------------------
    Name:		Reaction::GetCpi

    Input(3):	Temperature; Gas constant; The numbering of the component

    Function:	Get the specific heat at constant pressure for the given condition

    Return:		The component Cp for the given condition
 *---------------------------------------------------------------------------------*/

double Reaction::GetCpi(double T, double R, int SP, Array<double, 2> &Coeff0, Array<double, 2> &Coeff1)
{
    double Cpi = 0.0;

    if (T < 1000)
        Cpi = R * (Coeff0(0, SP) * pow(T, -2) + Coeff0(1, SP) * pow(T, -1) + Coeff0(2, SP) + Coeff0(3, SP) * T + Coeff0(4, SP) * pow(T, 2) + Coeff0(5, SP) * pow(T, 3) + Coeff0(6, SP) * pow(T, 4));
    else
        Cpi = R * (Coeff1(0, SP) * pow(T, -2) + Coeff1(1, SP) * pow(T, -1) + Coeff1(2, SP) + Coeff1(3, SP) * T + Coeff1(4, SP) * pow(T, 2) + Coeff1(5, SP) * pow(T, 3) + Coeff1(6, SP) * pow(T, 4));
    // if (SP == 0)
    //     cout << T << " " << R << " " << Coeff0(0, 0) << " " << Coeff0(1, 0) << " " << Coeff0(2, 0) << " " << Coeff0(3, 0) << endl;
    return Cpi;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::GetHi

    Input(3):	Temperature; Gas constant; The numbering of the component

    Function:	Get the enthalpy for the given condition

    Return:	The component enthalpy for the given condition
 *---------------------------------------------------------------------------------*/

double Reaction::GetHi(double T, double R, int SP, Array<double, 2> &Coeff0, Array<double, 2> &Coeff1)
{
    double Hi = 0.0;

    if (T < 1000)
        Hi = R * (-Coeff0(0, SP) * pow(T, -1) + Coeff0(1, SP) * log(T) + Coeff0(2, SP) * T + Coeff0(3, SP) * pow(T, 2) / 2 + Coeff0(4, SP) * pow(T, 3) / 3 + Coeff0(5, SP) * pow(T, 4) / 4 + Coeff0(6, SP) * pow(T, 5) / 5 + Coeff0(7, SP));
    else
        Hi = R * (-Coeff1(0, SP) * pow(T, -1) + Coeff1(1, SP) * log(T) + Coeff1(2, SP) * T + Coeff1(3, SP) * pow(T, 2) / 2 + Coeff1(4, SP) * pow(T, 3) / 3 + Coeff1(5, SP) * pow(T, 4) / 4 + Coeff1(6, SP) * pow(T, 5) / 5 + Coeff1(7, SP));

    return Hi;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::GetSi

    Input(3):	Temperature; Gas constant; The numbering of the component

    Function:	Get the entropy for the given condition

    Return:		The component entropy for the given condition
 *---------------------------------------------------------------------------------*/

double Reaction::GetSi(double T, double R, int SP, Array<double, 2> &Coeff0, Array<double, 2> &Coeff1)
{
    double Si = 0.0;

    if (T < 1000)
        Si = R * (-Coeff0(0, SP) * pow(T, -2) / 2 - Coeff0(1, SP) * pow(T, -1) + Coeff0(2, SP) * log(T) + Coeff0(3, SP) * T + Coeff0(4, SP) * pow(T, 2) / 2 + Coeff0(5, SP) * pow(T, 3) / 3 + Coeff0(6, SP) * pow(T, 4) / 4 + Coeff0(8, SP));
    else
        Si = R * (-Coeff1(0, SP) * pow(T, -2) / 2 - Coeff1(1, SP) * pow(T, -1) + Coeff1(2, SP) * log(T) + Coeff1(3, SP) * T + Coeff1(4, SP) * pow(T, 2) / 2 + Coeff1(5, SP) * pow(T, 3) / 3 + Coeff1(6, SP) * pow(T, 4) / 4 + Coeff1(8, SP));

    return Si;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::GetWi

    Input(0):	None

    Function:	Get the component mass production rate

    Return:		The component mass production rate
 *---------------------------------------------------------------------------------*/

Array<double, 4> Reaction::GetWi()
{
    return CMS;
}

/*---------------------------------------------------------------------------------
    Name:		Reaction::GetMD

    Input(0):	None

    Function:	Get the diagonal of the Jacobi matrix

    Return:		The diagonal of the Jacobi matrix
 *---------------------------------------------------------------------------------*/

Array<double, 5> Reaction::GetMD()
{
    return MD;
}

void Reaction::PackageUpdate(std::vector<int> &data, int meshnum, std::vector<int> &recvFrom, std::vector<int> &transferNmesh)
{
    int index = 0, start = 0, size = 0;
    for (int n = 0; n < recvFrom.size(); n++)
    {
        start = std::accumulate(transferNmesh.begin(), transferNmesh.begin() + recvFrom[n], 0);
        size = transferNmesh[recvFrom[n]];
        for (int i = 0; i < size; index++, i++)
            data[start++] = Nchem(index, 0, 0);
    }
}

void Reaction::UnpackageUpdate(std::vector<int> &data, std::vector<int> &transferIndex, int meshnum, std::vector<int> &sendTo, std::vector<int> &transferNmesh)
{
    int index = 0, start = 0, size = 0;
    int x = 0, y = 0, z = 0;
    for (int n = 0; n < sendTo.size(); n++)
    {
        start = std::accumulate(transferNmesh.begin(), transferNmesh.begin() + sendTo[n], 0);
        size = transferNmesh[sendTo[n]];
        for (int i = 0; i < size; index++, i++)
        {
            x = transferIndex[3 * index];
            y = transferIndex[3 * index + 1];
            z = transferIndex[3 * index + 2];
            Nchem(x, y, z) = data[start++];
        }
    }
}

void Reaction::PackagePrev(std::vector<int> &data, int i, int j, int k)
{
    data.push_back(Nchem(i, j, k));
}
void Reaction::UnpackagePrev(std::vector<int> &data, std::vector<int> &recvFrom, std::vector<int> &transferNmesh)
{
    int index = 0, start = 0, size = 0;
    for (int n = 0; n < recvFrom.size(); n++)
    {
        start = std::accumulate(transferNmesh.begin() + recvFrom[0], transferNmesh.begin() + recvFrom[n], 0);
        size = transferNmesh[recvFrom[n]];
        for (int i = 0; i < size; index++, i++)
            Nchem(index, 0, 0) = data[start++];
    }
}
void Reaction::UnpackagePrev(std::vector<int> &data, int meshnum, std::vector<int> &recvFrom, std::vector<int> &transferNmesh)
{
    // cout << "unpackage start\n";
    int index = 0, start = 0, size = 0;
    for (int n = 0; n < recvFrom.size(); n++)
    {
        start = std::accumulate(transferNmesh.begin(), transferNmesh.begin() + recvFrom[n], 0);
        size = transferNmesh[recvFrom[n]];
        for (int i = 0; i < size; index++, i++)
            Nchem(index, 0, 0) = data[start++];
    }
}

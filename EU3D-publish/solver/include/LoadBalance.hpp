#ifndef DLB_H
#define DLB_H
#include <vector>
#include <queue>
#include <tuple>
#include <mpi.h>
#include "Array.hpp"

using namespace ARRAY;
class DynamicLoadBalancer
{
public:
    /*parameter*/
    std::vector<std::tuple<int, int, int>> transferNchem; /* 1.send 2.receive 3.transfer amount */
    double LoadDegree = 0.0;
    double LoadDegreeBalance = 0.0;
    int NchemSumBalance = 0;
    std::vector<int> NchemTotalBalance;
    std::vector<int> sendTo;                     /* send to process line in transferNchem*/
    std::vector<int> recvFrom;                   /* recv from process line in transferNchem*/
    std::vector<int> transferIndex;              /* transfer mesh index of local process*/
    std::vector<double> prevFlowDataLocal;       /* previous flow data of local process*/
    std::vector<int> prevReactDataLocal;         /* previous Nchem data of local process*/
    std::vector<double> updateFlowDataLocal;     /* update flow data of local process*/
    int sentNmesh = 0;                           /* transfer mesh number of local process*/
    std::vector<int> transferMeshNum;            /* transfer mesh number of whole processes*/
    std::vector<int> transferNmesh;              /* reduce transfer mesh number of whole processes*/
    std::vector<double> transferPrevDataLocal;   /* local transfer data of whole processes */
    std::vector<double> transferPrevData;        /* transfer data of whole processes */
    std::vector<double> transferUpdateDataLocal; /* local transfer data of whole processes */
    std::vector<int> transferPrevNchem;          /* transfer Nchem of whole processes */
    std::vector<int> transferPrevNchemLocal;     /* local transfer Nchem of whole processes */
    std::vector<double> transferUpdateData;      /* transfer data of whole processes */
    std::vector<int> transferUpdateNchem;
    std::vector<int> transferUpdateNchemLocal;
    std::vector<double> recvUpdateFlowData;
    std::vector<double> recvPrevFlowData;
    std::vector<int> recvPrevReactData;

    /*function*/
    DynamicLoadBalancer() = default;
    void DLBPriorityQueue(Array<int, 1> &arr);
    int GetSentNchem(int element);
    int GetRecvNmesh(int element);
    void GetTransferMesh(Array<int, 3> &Nchem,int bc);
    ~DynamicLoadBalancer() { ; };
};
#endif
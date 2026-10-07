#include <LoadBalance.hpp>
#include <algorithm>
#include <numeric>
#include <Global.hpp>
#include <limits.h>
#include <tuple>
#include <unordered_map>
void DynamicLoadBalancer::DLBPriorityQueue(Array<int, 1> &arr)
{
    int n = arr.GetSize();
    int average = 0;
    int sum = 0;

    average = arr.AveValue();

    std::priority_queue<std::pair<int, int>> maxHeap;
    std::priority_queue<std::pair<int, int>> minHeap;

    // Split the array into two parts: greater than or equal to average, and less than average
    for (int i = 0; i < n; i++)
    {
        if (arr(i) >= average)
        {
            maxHeap.push({arr(i), i});
        }
        else
        {
            minHeap.push({arr(i), i});
        }
    }

    // Perform transfer until the difference of the maximum value is 0 or the difference of the minimum value is 0
    while (!maxHeap.empty())
    {
        int sender = maxHeap.top().second;
        int diff = maxHeap.top().first - average;
        maxHeap.pop();

        while (!minHeap.empty() && diff > 0)
        {
            int receiver = minHeap.top().second;
            int surplus = average - minHeap.top().first;
            minHeap.pop();

            int transferAmount = std::min(diff, surplus);
            arr(sender) -= transferAmount;
            arr(receiver) += transferAmount;

            transferNchem.push_back(std::make_tuple(sender, receiver, transferAmount));

            diff -= transferAmount;

            if (arr(receiver) < average)
            {
                minHeap.push({arr(receiver), receiver});
            }
        }
    }
}

int DynamicLoadBalancer::GetSentNchem(int element)
{
    sendTo.clear();
    int totalSentAmount = 0;
    int line = 0;
    for (const auto &transfer : transferNchem)
    {
        int sender = std::get<0>(transfer);
        int receiver = std::get<1>(transfer);
        int amount = std::get<2>(transfer);

        if (sender == element)
        {
            sendTo.push_back(line);
            totalSentAmount += amount;
        }
        line++;
    }
    return totalSentAmount;
}

int DynamicLoadBalancer::GetRecvNmesh(int element)
{
    recvFrom.clear();
    int totalRecvAmount = 0;
    int line = 0;
    for (const auto &transfer : transferNchem)
    {
        int sender = std::get<0>(transfer);
        int receiver = std::get<1>(transfer);

        if (receiver == element)
        {
            recvFrom.push_back(line);
            totalRecvAmount += transferNmesh[line];
        }
        line++;
    }
    return totalRecvAmount;
}

void DynamicLoadBalancer::GetTransferMesh(Array<int, 3> &Nchem, int bc)
{
    int ni = Nchem.GetNi() - 2 * bc;
    int nj = Nchem.GetNj() - 2 * bc;
    int nk = Nchem.GetNk() - 2 * bc;
    std::vector<int> nchem(Nchem.Getbuf(), Nchem.Getbuf() + Nchem.GetSize());
    std::vector<int> index(nchem.size());
    std::iota(index.begin(), index.end(), 0);

    // Global::quickSort(nchem, index, 0, nchem.size() - 1);

    std::vector<int> targetSums;
    for (int line : sendTo)
        targetSums.push_back(std::get<2>(transferNchem[line]));

    std::vector<std::vector<int>> packNchem;
    std::vector<std::vector<int>> packIndices;
    packNchem.resize(targetSums.size());
    packIndices.resize(targetSums.size());

    // for (int n = nchem.size() - 1; n >= 0; n--)
    for (int n = 0; n < nchem.size(); n++)
    {
        std::tuple<int, int, int> arrayIndex = Nchem.Get3DIndices(index[n]);
        if (std::get<0>(arrayIndex) >= bc && std::get<1>(arrayIndex) >= bc && std::get<2>(arrayIndex) >= bc &&
            std::get<0>(arrayIndex) < ni + bc && std::get<1>(arrayIndex) < nj + bc && std::get<2>(arrayIndex) < nk + bc)
        {
            int num = nchem[n];
            int minDiff = INT_MAX;
            int targetIndex = -1;

            // Find the position closest to the target sum.
            for (int i = 0; i < targetSums.size(); i++)
            {
                int currentSum = std::accumulate(packNchem[i].begin(), packNchem[i].end(), 0);
                int currentDiff = std::abs(targetSums[i] - currentSum);

                if (currentDiff < minDiff && currentSum + num <= targetSums[i])
                {
                    minDiff = currentDiff;
                    targetIndex = i;
                }
            }

            if (targetIndex != -1)
            {
                packNchem[targetIndex].push_back(num);
                packIndices[targetIndex].push_back(std::get<0>(arrayIndex));
                packIndices[targetIndex].push_back(std::get<1>(arrayIndex));
                packIndices[targetIndex].push_back(std::get<2>(arrayIndex));
                transferMeshNum[sendTo[targetIndex]]++;
            }
        }
    }

    for (const auto &row : packIndices)
        transferIndex.insert(transferIndex.end(), row.begin(), row.end());

    // for (int i = 0; i < packNchem.size(); i++)
    // {
    //     std::cout << "Array " << i + 1 << ": ";
    //     for (int num : packNchem[i])
    //     {
    //         std::cout << num << " ";
    //     }
    //     std::cout << std::endl;

    //     std::cout << "Indices " << i + 1 << ": ";
    //     for (int index : packIndices[i])
    //     {
    //         std::cout << index << " ";
    //     }
    //     std::cout << std::endl;
    // }
}

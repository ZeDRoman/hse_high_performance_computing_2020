#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>

double T = 0.1;

double K = 1;
//double h = 0.02;
//double dt = 0.0002;
//size_t pieces = 1 / h  + 1;
size_t pieces = 51;
double h = 1.0 / (pieces - 1);
double dt = h * h  / 2;
size_t point_step = 11;

double computeCorrect(double x, double t) {
        double sum = 0;
        int M = 10000;
        for (int m = 0; m < M; ++m) {
            sum += exp(-1.0 * (M_PI * M_PI) * (2 * m + 1) * (2 * m + 1) * t) * sin(M_PI * (2 * m + 1) * x) / (2 * m + 1.0);
        }
        return 4 / M_PI * sum;
    }

double formula(double left, double right, double val) {
    return val + K * dt / std::pow(h, 2) * (left + right  - 2 * val);
}

std::vector<double> perform_step(std::vector<double>& values, double left, double right, int rank) {
    std::vector<double> new_values(values.size(), 0);
    for (int i = 0; i < values.size(); ++i) {
        auto l = i > 0 ? values[i - 1] : left;
        auto r = i < values.size() - 1 ? values[i + 1]  : right;
        new_values[i] = formula(l, r, values[i]);
    }
    return new_values;
}

void get_results(int rank, int process_count, std::vector<double>& values) {
    if (rank == 0) {
        std::vector<double> full_result(pieces);
        for (int i = 0; i < values.size(); ++i) {
            full_result[i] = values[i];
        }
        int size = values.size();
        for (int i = 1; i < process_count; ++i) {
            std::size_t pieces_amount = pieces / process_count;
            if (i < pieces % process_count) {
                pieces_amount += 1;
            }
            auto ec = MPI_Recv(full_result.data() + size, pieces_amount, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD,NULL);
            size += pieces_amount;
            if (ec != MPI_SUCCESS) {
                MPI_Abort(MPI_COMM_WORLD, ec);
            }
        }

        for (int i = 0; i < pieces; i += static_cast<int>(pieces / (point_step - 1))) {
            std::cout << full_result[i] << " ";
        }
        std::cout << "\n";
    }
    else {
        MPI_Send(values.data(), values.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void process() {
    int rank = 0;
    int process_count = 0;
    double local_time = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&rank));
    MPI_Comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&process_count));
    std::size_t pieces_amount = pieces / process_count;
    if (rank < pieces % process_count) {
        pieces_amount += 1;
    }
    std::vector<double> values(pieces_amount, 1);
    double prev = 1;
    double next = 1;
    if (rank == process_count - 1) {
        next = 0;
    }
    if (rank == 0) {
        prev = 0;
    }

    while (local_time < T) {
        values = perform_step(values, prev, next, rank);
        local_time += dt;
        if (rank != 0) {
            MPI_Recv(&prev, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
            MPI_Send(&values[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
        }
        if (rank != process_count - 1) {
            MPI_Send(&values.back(), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD);
            MPI_Recv(&next, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        }
    }
    get_results(rank, process_count, values);
}


int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    process();
    MPI_Finalize();
}

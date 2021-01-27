#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <mpi.h>


//size_t pieces = 10001;
//double h = 1.0 / (pieces - 1);
//double dt = h * h  / 2;
class config {
public:
    config() = default;
    config(double T, int pieces) : FullTime(T) {
        points = pieces + 1;
        piece_width = 1.0 / points;
        dt = piece_width * piece_width / 2;
        need_ans = false;
    };
    double FullTime = 0.1;
    double K = 1;
    double piece_width = 0.02;
    double dt = 0.0002;
    int points = 1 / piece_width  + 1;
    int points_step = 11;
    bool need_ans = true;
};

double correct_answer(double x, double t) {
    double sum = 0;
    int M = 10000;
    for (int m = 0; m < M; ++m) {
        sum += exp(-1.0 * (M_PI * M_PI) * (2 * m + 1) * (2 * m + 1) * t) * sin(M_PI * (2 * m + 1) * x) / (2 * m + 1.0);
    }
    return 4 / M_PI * sum;
}

double formula(double left, double right, double val, config conf) {
    return val + conf.K * conf.dt / std::pow(conf.piece_width, 2) * (left + right  - 2 * val);
}

std::vector<double> perform_step(std::vector<double>& values, double left, double right, int rank, int process_count, config conf) {
    std::vector<double> new_values(values.size(), 0);
    for (int i = 1; i < values.size() - 1; ++i) {
        auto l = values[i - 1];
        auto r = values[i + 1];
        new_values[i] = formula(l, r, values[i], conf);
    }
    if (rank != 0) {
        MPI_Recv(&left, 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    if (rank != process_count - 1) {
        MPI_Recv(&right, 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    auto l = left;
    auto r = 0 < values.size() - 1 ? values[1] : right;
    new_values[0] = formula(l, r, values[0], conf);
    if (values.size() != 1) {
        new_values[values.size() - 1] = formula(values[values.size() - 2], right, values[values.size() - 1], conf);
    }
    return new_values;
}

void get_results(int rank, int process_count, std::vector<double>& values, config conf) {
    if (rank == 0) {
        std::vector<double> full_result(conf.points);
        for (int i = 0; i < values.size(); ++i) {
            full_result[i] = values[i];
        }
        int size = values.size();
        for (int i = 1; i < process_count; ++i) {
            std::size_t pieces_amount = conf.points / process_count;
            if (i < conf.points % process_count) {
                pieces_amount += 1;
            }
            MPI_Recv(full_result.data() + size, pieces_amount, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD,NULL);
            size += pieces_amount;
        }

        if (conf.need_ans) {
            int j = 1;
            for (int i = 0; i < conf.points; i += static_cast<int>(conf.points / (conf.points_step - 1))) {
                std::cout << "point " << j <<
                          "; result_value: " << full_result[i] <<
                          "; formula_value: " << correct_answer(i * conf.piece_width, conf.FullTime) <<
                          "; difference: " << full_result[i] - correct_answer(i * conf.piece_width, conf.FullTime) <<
                          "\n";
                j++;
            }
            std::cout << "\n";
        }
    }
    else {
        MPI_Send(values.data(), values.size(), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}

void process(config conf) {
    int rank = 0;
    int process_count = 0;
    double local_time = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&rank));
    MPI_Comm_size(MPI_COMM_WORLD, reinterpret_cast<int *>(&process_count));
    std::size_t points_amount = conf.points / process_count;
    if (rank < conf.points % process_count) {
        points_amount += 1;
    }
    std::vector<double> values(points_amount, 1);
    double prev = 1;
    double next = 1;
    if (rank == process_count - 1) {
        next = 0;
    }
    if (rank == 0) {
        prev = 0;
    }

    MPI_Request req1, req2;
    while (local_time < conf.FullTime) {
        if (rank != 0) {
            MPI_Isend(&values[0], 1, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &req1);
            MPI_Request_free(&req1);
        }
        if (rank != process_count - 1) {
            MPI_Isend(&values.back(), 1, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &req2);
            MPI_Request_free(&req2);
        }
        values = perform_step(values, prev, next, rank, process_count, conf);
        local_time += conf.dt;
    }
    get_results(rank, process_count, values, conf);
}

void run_test(int argc, char** argv, config conf) {
    MPI_Init(&argc, &argv);
    process(conf);
    MPI_Finalize();
}

void measure_times(int argc, char** argv) {
    double T = 0.0001;
    int N = 50000;
    config conf(T, N);
    int rank = 0;
    const auto begin = MPI_Wtime();
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, reinterpret_cast<int *>(&rank));
    process(conf);
    MPI_Finalize();
    const auto dur = MPI_Wtime() - begin;
    if (rank == 0)
        std::cout << "Finished in " << dur << " seconds" << '\n';
}

int main(int argc, char **argv)
{
    config default_config;
    run_test(argc, argv, default_config);
}

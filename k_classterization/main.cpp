#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <tuple>
#include <chrono>


constexpr int TEST_CASES = 5;
const std::string DATA_FOLDER = "/home/ilgovskiy/CLionProjects/k_clasterization/data/";
const int MAX_THREADS = omp_get_max_threads();

struct Point{
    Point(): x(0), y(0), cluster(0) {}
    Point(double _x, double _y): x(_x), y(_y) {}
    double x;
    double y;
    int cluster = -1;

    Point & operator +=(const Point &b) {
        x += b.x;
        y += b.y;
        return *this;
    }

    Point & operator /=(const double &val) {
        x /= val;
        y /= val;
        return *this;
    }

    bool operator ==(const Point &b) {
        return std::tie(x, y) == std::tie(b.x, b.y);
    }

    [[nodiscard]] double distance(Point b) const {
        return sqrt(pow(x - b.x, 2) + pow(y - b.y, 2));
    }
};

struct TaskData{
    std::vector<Point> points;
    int clusters_amount{};
    int points_amount{};
    std::vector<Point> centres;
};

void init_centers(TaskData& data) {
    for (int i = 0; i < data.clusters_amount; ++i) {
        data.centres.emplace_back(data.points[i % data.points_amount]);
        data.centres[i].cluster = i;
    }
}

void redefine_cluster(Point& point, std::vector<Point>& centres) {
    if (centres.empty()) {
        throw std::invalid_argument( "centres is empty" );;
    }
    point.cluster = 0;
    double distance = point.distance(centres[0]);
    for (size_t i = 1; i < centres.size(); ++i) {
        double new_distance = point.distance(centres[i]);
        if (new_distance < distance) {
            point.cluster = i;
            distance = new_distance;
        }
    }
}

bool redefine_centres(TaskData& data) {
    std::vector<Point> new_centers(data.clusters_amount);
    std::vector<int> cluster_counters(data.clusters_amount);

    for (auto p: data.points) {
        new_centers[p.cluster] += p;
        cluster_counters[p.cluster] += 1;
    }
    bool changed = false;
    for (size_t i = 0; i < data.clusters_amount; ++i) {
        new_centers[i] /= cluster_counters[i];
        if (!(data.centres[i] == new_centers[i])) {
            data.centres[i] = new_centers[i];
            changed = true;
        }
    }
    return changed;
}

void solve(TaskData& data) {
    int a = 0;
    while (true)
    {
        std::cout << a++ << "\n";
        int points_amount = data.points_amount;
        #pragma omp parallel for shared(data) schedule(guided)
        for (size_t i = 0; i < points_amount; ++i) {
            redefine_cluster(data.points[i], data.centres);
        }
        if (!redefine_centres(data)) {
            break;
        }
    }
}


long measure_test_time(TaskData data) {
    auto t1 = std::chrono::high_resolution_clock::now();
    solve(data);
    auto t2 = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>( t2 - t1 ).count();

    return duration;
}

void print_result_to_file(const std::string& output_file, const TaskData& data) {
    auto output = std::fstream(output_file, std::fstream::out);
    for (size_t i = 0; i < data.points_amount; ++i) {
        output << data.points[i].x << ", " << data.points[i].y << ", " << data.points[i].cluster << "\n";
    }
    output.close();
}

TaskData prepare_data(const std::string& input_file) {
    auto input = std::fstream{input_file};
    TaskData data;
    input >> data.points_amount >> data.clusters_amount;
    double x, y;
    for (int i = 0; i < data.points_amount; ++i)
    {
        input >> x >> y;
        data.points.emplace_back(x,y);
    }
    init_centers(data);
    return data;
}

void run_tests()
{
    auto output = std::fstream(DATA_FOLDER + "time_data.out", std::fstream::out);

    for (int i = 1; i <= TEST_CASES; ++i) {
        for (size_t thread_num = MAX_THREADS; thread_num <= MAX_THREADS; ++thread_num) {
        omp_set_num_threads(thread_num);
            TaskData data = prepare_data(DATA_FOLDER + "data" + std::to_string(i) + ".in");
            auto duration = measure_test_time(data);
            output << thread_num << ", " << i << ", " << duration << "\n";
        }
        output.flush();
    }
    output.close();
}

void get_answers()
{
    for (int i = 1; i <= TEST_CASES; ++i) {
        TaskData data = prepare_data(DATA_FOLDER + "data" + std::to_string(i) + ".in");
        solve(data);
        print_result_to_file(DATA_FOLDER + "data" + std::to_string(i) + ".out", data);
    }
}


int main()
{
    get_answers();
    run_tests();
}

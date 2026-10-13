// компілятор: g++ (GCC) 16.2.1
/*
Вхідні файли: data/data_*_A.txt та data/data_*_B.txt, дe * це розмір з kSizes.
 1 рядок: n - кількість чисел у файлі
 2 рядок: n невід'ємних цілих чисел у неспадному порядку розділених пробілами
dir за замовчуванням data
*/
#include <algorithm>
#include <chrono>
#include <cmath>
#include <execution>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <vector>

using namespace std;
namespace fs = std::filesystem;

using Seq = vector<int>;

static const vector<size_t> kSizes = {100'000, 1'000'000, 5'000'000, 20'000'000};
static const int kRepeats = 15;

// -----------------------------------------------------------------
string fileName(const string &dir, size_t n, char which) {
  return dir + "/data_" + to_string(n) + "_" + which + ".txt";
}

void writeSequence(const string &path, const Seq &s) {
  ofstream out(path);
  out << s.size() << '\n';
  for (int x : s)
    out << x << ' ';
  out << '\n';
}

Seq readSequence(const string &path) {
  ifstream in(path);
  size_t n;
  in >> n;
  Seq s(n);
  for (auto &x : s)
    in >> x;
  return s;
}

Seq randomSorted(size_t n, mt19937 &gen) {
  uniform_int_distribution<int> dist(0, 1'000'000'000);
  Seq s(n);
  for (auto &x : s)
    x = dist(gen);
  sort(s.begin(), s.end());
  return s;
}

void generateData(const string &dir) {
  fs::create_directories(dir);
  random_device rd;
  mt19937 gen(rd());
  for (size_t n : kSizes) {
    writeSequence(fileName(dir, n, 'A'), randomSorted(n, gen));
    writeSequence(fileName(dir, n, 'B'), randomSorted(n, gen));
    cout << "generated n = " << n << endl;
  }
}
// -----------------------------------------------------------------
using Clock = chrono::steady_clock;

double measureMs(const function<void()> &f) {
  vector<double> t;
  for (int i = 0; i < kRepeats; ++i) {
    auto t1 = Clock::now();
    f();
    auto t2 = Clock::now();
    t.push_back(chrono::duration<double, milli>(t2 - t1).count());
  }
  sort(t.begin(), t.end());
  return t[t.size() / 2];
}

// -------------------------------------------------------------------
template <class It1, class It2, class Out, class Comp>
void parallelMerge(It1 a, It1 aEnd, It2 b, It2 bEnd, Out out, unsigned K, Comp comp) {
  const size_t na = aEnd - a, nb = bEnd - b;
  vector<size_t> ai(K + 1), bj(K + 1);
  ai[0] = bj[0] = 0;
  ai[K] = na;
  bj[K] = nb;
  for (unsigned k = 1; k < K; ++k) {
    ai[k] = na * k / K;
    bj[k] = ai[k] < na ? lower_bound(b, bEnd, *(a + ai[k]), comp) - b : nb;
  }
  vector<jthread> threads;
  threads.reserve(K);
  for (unsigned k = 0; k < K; ++k) {
    threads.emplace_back([=] {
      merge(a + ai[k], a + ai[k + 1], b + bj[k], b + bj[k + 1],
            out + (ai[k] + bj[k]), comp);
    });
  }
}
// --------------------------------------------------------------
struct Data {
  Seq a, b, ref;
};

auto lightComp = [](int x, int y) { return x < y; };

auto heavyComp = [](int x, int y) { return sqrt((double)x) < sqrt((double)y); };

void printRow(const string &name, double ms) {
  cout << "  " << left << setw(34) << name << right << setw(10) << fixed << setprecision(2) << ms << " ms" << endl;
}

template <class Pol, class Comp>
void libraryRow(const string &name, Pol &&p, const Data &d, Seq &out, Comp comp) {
  printRow(name, measureMs([&] {
    merge(p, d.a.begin(), d.a.end(), d.b.begin(), d.b.end(), out.begin(), comp);
  }));
}

template <class Comp>
void experiment(const string &title, const Data &d, Seq &out, Comp comp) {
  cout << "Library merge, " << title << ":" << endl;
  printRow("no policy", measureMs([&] {
    merge(d.a.begin(), d.a.end(), d.b.begin(), d.b.end(), out.begin(), comp);
  }));
  libraryRow("seq", execution::seq, d, out, comp);
  libraryRow("unseq", execution::unseq, d, out, comp);
  libraryRow("par", execution::par, d, out, comp);
  libraryRow("par_unseq", execution::par_unseq, d, out, comp);
}

template <class Comp>
void experimentOwnParallel(const string &title, const Data &d, Seq &out, Comp comp) {
  const unsigned hw = thread::hardware_concurrency();
  const vector<unsigned> ks = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128};
  cout << "Own parallel merge, " << title << " (hardware threads = " << hw << "):" << endl;
  cout << "  " << setw(5) << "K" << setw(12) << "time, ms" << setw(10) << "speedup"
       << setw(9) << "correct" << endl;
  double t1 = 0, best = 1e300;
  unsigned bestK = 1;
  for (unsigned K : ks) {
    double ms = measureMs([&] {
      parallelMerge(d.a.begin(), d.a.end(), d.b.begin(), d.b.end(), out.begin(), K, comp);
    });
    bool ok = (out == d.ref);
    if (K == 1)
      t1 = ms;
    if (ms < best)
      best = ms, bestK = K;
    cout << "  " << setw(5) << K << setw(12) << fixed << setprecision(2) << ms
         << setw(10) << t1 / ms << setw(9) << (ok ? "yes" : "NO") << endl;
  }
  cout << "  best K = " << bestK << " (" << best << " ms); K / hardware threads = "
       << setprecision(2) << (double)bestK / hw << endl;
}

void runExperiments(const string &dir) {
  for (size_t n : kSizes) {
    Data d;
    d.a = readSequence(fileName(dir, n, 'A'));
    d.b = readSequence(fileName(dir, n, 'B'));
    d.ref.resize(d.a.size() + d.b.size());
    merge(d.a.begin(), d.a.end(), d.b.begin(), d.b.end(), d.ref.begin());
    Seq out(d.ref.size());
    cout << "\n===== n = " << n << " + " << n << " elements =====" << endl;
    experiment("operator<", d, out, lightComp);
    experiment("heavy predicate (sqrt)", d, out, heavyComp);
    experimentOwnParallel("operator<", d, out, lightComp);
  }
}

int main(int argc, char **argv) {
  string mode = argc > 1 ? argv[1] : "run";
  string dir = argc > 2 ? argv[2] : "data";
  if (mode == "gen")
    generateData(dir);
  else
    runExperiments(dir);
  return 0;
}

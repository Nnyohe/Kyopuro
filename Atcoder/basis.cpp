#include <bits/stdc++.h>
using namespace std;
#include <atcoder/all>
#include <cassert>
#include <functional>
#include <random>
using namespace atcoder;
using ll = long long;
using ld = long double;
using Graph = vector<vector<int>>;
using vi = vector<int>;
using vl = vector<long>;
using vll = vector<long long>;
using vb = vector<bool>;
using vvi = vector<vi>;
using vvl = vector<vl>;
using vvb = vector<vb>;
using vvvb = vector<vvb>;
using vvll = vector<vll>;
using vvvll = vector<vvll>;
using vvvvll = vector<vvvll>;
using vvvvvll = vector<vvvvll>;
using vd = vector<double>;
using vvd = vector<vd>;
using vvvd = vector<vvd>;
using vld = vector<long double>;
using vvld = vector<vld>;
using vvvld = vector<vvld>;
using vc = vector<char>;
using vvc = vector<vc>;
using lll = __int128_t;
using vs = vector<string>;
using pii = pair<long long, long long>;
using mint = modint1000000007;
#define reps(i, a, n) for (ll i = (a); i < (ll)(n); i++)
#define rep(i, n) for (ll i = (0); i < (ll)(n); i++)
#define rrep(i, n) for (ll i = (1); i < (ll)(n + 1); i++)
#define repd(i, n) for (ll i = n - 1; i >= 0; i--)
#define rrepd(i, n) for (ll i = n; i >= 1; i--)
#define ALL(n) n.begin(), n.end()
#define rALL(n) n.rbegin(), n.rend()
#define fore(i, a) for (auto &i : a)
#define IN(a, x, b) (a <= x && x < b)
#define IN(a, x, b) (a <= x && x < b)
#define INIT                          \
    std::ios::sync_with_stdio(false); \
    std::cin.tie(0);

template <typename T>
// [0,M)についての階上を求める
vector<T> KAI(int M) {
    vector<T> kai(M, 1);
    rep(i, M - 1) { kai[i + 1] = kai[i] * (i + 1); }
    return kai;
}
template <typename T>
vector<T> DIV(int M) {
    vector<T> kai = KAI<T>(M), div(M, 1);

    rep(i, M - 1) { div[i + 1] = 1 / kai[i + 1]; }
    return div;
}

long long Power(long long a, long long b, long long m) {
    long long p = a, Answer = 1;
    p %= m;
    for (int i = 0; i < 63; i++) {
        ll wari = (1LL << i);
        if ((b / wari) % 2 == 1) {
            Answer %= m;
            Answer = (Answer * p) % m;
            // 「a の 2^i 乗」が掛けられるとき
        }
        ll t = p % m;
        p = (t * t) % m;
        // cout << p << endl;
    }
    return Answer;
}

// a ÷ b を m で割った余りを返す関数
long long Division(long long a, long long b, long long m) {
    return (a * Power(b, m - 2, m)) % m;
}
template <class T>
void output(T &W, bool x) {
    cout << W;
    if (!x)
        cout << ' ';
    else
        cout << endl;
    return;
}
// sは改行するか否かを表す
template <class T>
void output(vector<T> &W, bool s) {
    rep(i, W.size()) { output(W[i], ((i == W.size() - 1) || s)); }
    return;
}
// sは改行するか否かを表す
template <class T>
void output(vector<vector<T>> &W, bool s) {
    rep(i, W.size()) { output(W[i], s); }
    return;
}
template <class T>
T vectorsum(vector<T> &W, int a, int b) {
    return accumulate(W.begin() + a, W.end() + b, (T)0);
}
template <class T>
T vectorsum(vector<T> &W) {
    int b = W.size();
    return accumulate(ALL(W), (T)0);
}
template <class T>
inline T CHMAX(T &a, const T b) {
    return a = (a < b) ? b : a;
}
template <class T>
inline T CHMIN(T &a, const T b) {
    return a = (a > b) ? b : a;
}
template <class T>
void input(T &W) {
    cin >> W;
    return;
}

template <class T>
void input(vector<T> &W) {
    for (auto &u : W) input(u);
    return;
}
template <class T, class TT>
void add(T &W, TT &a) {
    W += a;
    return;
}
template <class T>
void add(vector<T> &W, vector<T> &a) {
    rep(i, W.size()) add(W[i], a[i]);
}
template <class T>
void add(T &W, T &a) {
    W += a;
}
template <class T, class TT>
void add(vector<T> &W, TT a) {
    for (auto &u : W) add(u, a);
    return;
}

vll dx = {1, -1, 0, 0}, dy = {0, 0, 1, -1};

void Yes(bool b) {
    if (b)
        cout << "Yes" << '\n';
    else
        cout << "No" << '\n';
}
struct TRI {
    ll a;
    ll b;
    ll c;
    ll d;
};
bool operator>(const TRI &r, const TRI &l) {
    return (r.a > l.a | (r.a == l.a & r.b > l.b) |
            (r.a == l.a & r.b == l.b & r.c > l.c));
}
bool operator<(const TRI &r, const TRI &l) {
    return (r.a < l.a | (r.a == l.a & r.b < l.b) |
            (r.a == l.a & r.b == l.b & r.c < l.c));
}
struct TRK {
    ll a;
    ll b;
    ll c;
};
bool operator>(const TRK &r, const TRK &l) {
    return (r.a > l.a | (r.a == l.a & r.b > l.b) |
            (r.a == l.a & r.b == l.b & r.c > l.c));
}
bool operator<(const TRK &r, const TRK &l) {
    return (r.a < l.a | (r.a == l.a & r.b < l.b) |
            (r.a == l.a & r.b == l.b & r.c < l.c));
}

void solve() {
    cin.tie(0)->sync_with_stdio(0);
    cout << fixed << setprecision(20);
    ll a = 0, b = 0;
    ll a2, b2, c2;
    ll a1 = 0, b1 = 0;
    ll c = 0, c1 = 0;
    ll p = 0;
    ll N, M, K, L;
    ll t;
    ll h, w;
    string S, T;
}
int main() {
    ll a = 0, b = 0;
    ll a2, b2, c2;
    ll a1 = 0, b1 = 0;
    ll c = 0, c1;
    ll p = 0;
    ll N, M;
    ll t;
    ll K;
    ll h, w;
    t = 1;
    //  cin >> t;
    // cout << -1 << '\n';
    rep(_, t) solve();
}

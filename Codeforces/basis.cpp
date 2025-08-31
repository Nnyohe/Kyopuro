#include <bits/stdc++.h>
using namespace std;
#include <cassert>
#include <functional>
using ll = long long;
using ld = long double;
using Graph = vector<vector<int>>;
using vi = vector<int>;
using vl = vector<long long>;
using vll = vector<ll>;
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

#define mp make_pair
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
        cout << '\n';
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

const double PI = acos(-1.0L);
const long double EPS = 1e-10;
const double INF = 1e18;

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

#include <bits/stdc++.h>
using namespace std;
// 実装はUnion by sizeで行っている

class UnionFind {
   public:
    UnionFind() = default;

    /// @param n 要素数
    explicit UnionFind(size_t n) : m_parentsOrSize(n, -1), N(n) {}
    /// @brief 頂点 i の root のインデックスを返します。
    /// @param i 調べる頂点のインデックス
    /// @return 頂点 i の root のインデックス
    int leader(int i) {
        if (m_parentsOrSize[i] < 0) {
            return i;
        }
        const int root = leader(m_parentsOrSize[i]);
        // 経路圧縮
        return (m_parentsOrSize[i] = root);
    }

    /// @param w (b の重み) - (a の重み)
    /// a を含むグループと b を含むグループを併合する
    // グループが一致している場合何もしない
    void merge(int a, int b) {
        a = leader(a);
        b = leader(b);

        if (a != b) {
            // union by size (小さいほうが子になる）
            if (-m_parentsOrSize[a] < -m_parentsOrSize[b]) {
                std::swap(a, b);
            }
            m_parentsOrSize[a] += m_parentsOrSize[b];
            m_parentsOrSize[b] = a;
        }
    }

    /// @brief a と b が同じグループに属すかを返します。
    /// @param a 一方のインデックス
    /// @param b 他方のインデックス
    /// @return a と b が同じグループに属す場合 true, それ以外の場合は false
    /// a と b が同じグループに属すかを返す
    bool same(int a, int b) { return (leader(a) == leader(b)); }

    /// @brief i が属するグループの要素数を返します。
    /// @param i インデックス
    /// @return i が属するグループの要素数
    int size(int i) { return -m_parentsOrSize[leader(i)]; }
    vector<vector<int>> Groups() {
        vector<vector<int>> G;
        int sum = 0;
        vector<int> number(N, -1);
        for (int i = 0; i < N; i++) {
            int a = leader(i);
            if (number[a] == -1) {
                number[a] = sum;
                G.emplace_back(vector<int>{});
                G[sum].emplace_back(i);
                sum++;
            } else {
                G[number[i]].emplace_back(i);
            }
        }
        return G;
    }

   private:
    // m_parentsOrSize[i] は i の 親,
    // ただし root の場合は (-1 * そのグループに属する要素数)
    int N;
    std::vector<int> m_parentsOrSize;
};
struct Mo {
    int n;
    vector<pair<int, int>> lr;

    explicit Mo(int n) : n(n) {}

    void add(int l, int r) { /* [l, r) */ lr.emplace_back(l, r); }

    template <typename AL, typename AR, typename EL, typename ER, typename O>
    void build(const AL &add_left, const AR &add_right, const EL &erase_left,
               const ER &erase_right, const O &out) {
        int q = (int)lr.size();
        int bs = n / min<int>(n, sqrt(q));
        vector<int> ord(q);
        iota(begin(ord), end(ord), 0);
        sort(begin(ord), end(ord), [&](int a, int b) {
            int ablock = lr[a].first / bs, bblock = lr[b].first / bs;
            if (ablock != bblock) return ablock < bblock;
            return (ablock & 1) ? lr[a].second > lr[b].second
                                : lr[a].second < lr[b].second;
        });
        int l = 0, r = 0;
        for (auto idx : ord) {
            while (l > lr[idx].first) add_left(--l);
            while (r < lr[idx].second) add_right(r++);
            while (l < lr[idx].first) erase_left(l++);
            while (r > lr[idx].second) erase_right(--r);
            out(idx);
        }
    }

    template <typename A, typename E, typename O>
    void build(const A &add, const E &erase, const O &out) {
        build(add, add, erase, erase, out);
    }
};

vector<bool> Eratosthenes(int N) {
    vector<bool> R(N + 1, true);

    R[0] = R[1] = false;

    for (int p = 2; p <= N; ++p) {
        if (!R[p]) continue;

        for (int q = p * 2; q <= N; q += p) {
            R[q] = false;
        }
    }

    return R;
}
template <typename T, typename E>
struct LazySegmentTree {
   private:
    int n{}, sz{}, height{};
    vector<T> data;
    vector<E> lazy;
    T (*op)(T, T);
    T (*e)();
    T (*mapping)(E, T);
    E (*composition)(E, E);
    E (*id)();

    void update(int k) { data[k] = op(data[2 * k], data[2 * k + 1]); }

    void all_apply(int k, const E x) {
        data[k] = mapping(x, data[k]);
        if (k < sz) lazy[k] = composition(x, lazy[k]);
    }

    void propagate(int k) {
        if (lazy[k] != id()) {
            all_apply(2 * k, lazy[k]);
            all_apply(2 * k + 1, lazy[k]);
            lazy[k] = id();
        }
    }

   public:
    LazySegmentTree() = default;
    explicit LazySegmentTree(int n, T (*op)(T, T), T (*e)(), T (*mapping)(E, T),
                             E (*composition)(E, E), E (*id)())
        : n(n),
          op(op),
          mapping(mapping),
          composition(composition),
          e(e),
          id(id) {
        sz = 1;
        height = 0;
        while (sz < n) sz <<= 1, height++;
        data = vector<T>(2 * sz, e());
        lazy = vector<E>(sz, id());
    }

    explicit LazySegmentTree(const vector<T> &v, T (*op)(T, T), T (*e)(),
                             T (*mapping)(E, T), E (*composition)(E, E),
                             E (*id)())
        : LazySegmentTree(v.size(), op, e, mapping, composition, id) {
        build(v);
    }

    void build(const vector<T> &v) {
        assert(n == (int)v.size());
        for (int k = 0; k < n; k++) data[k + sz] = v[k];
        for (int k = sz - 1; k > 0; k--) update(k);
    }

    void set(int k, const T x) {
        assert(0 <= k && k < n);
        k += sz;
        for (int i = height; i > 0; i--) propagate(k >> i);
        data[k] = x;
        for (int i = 1; i <= height; i++) update(k >> i);
    }

    T get(int k) {
        assert(0 <= k && k < n);
        k += sz;
        for (int i = height; i > 0; i--) propagate(k >> i);
        return data[k];
    }

    T operator[](int k) { return get(k); }

    T prod(int l, int r) {
        assert(0 <= l && l <= r && r <= n);
        if (l >= r) return e();
        l += sz;
        r += sz;
        // 更新する区間を部分的に含んだ区間においてトップダウンで子ノードに伝搬させながらdataの値を更新
        for (int i = height; i > 0; i--) {
            if (((l >> i) << i) != l) propagate(l >> i);
            if (((r >> i) << i) != r) propagate((r - 1) >> i);
        }
        T L = e(), R = e();
        // 値をチェックする区間のdataの値をチェック
        for (; l < r; l >>= 1, r >>= 1) {
            if (l & 1) L = op(L, data[l++]);
            if (r & 1) R = op(data[--r], R);
        }
        return op(L, R);
    }

    T all_prod() const { return data[1]; }

    void apply(int k, const E &x) {
        assert(0 <= k && k < n);
        k += sz;
        for (int i = height; i > 0; i--) propagate(k >> i);
        data[k] = mapping(x, data[k]);
        for (int i = 1; i <= height; i++) update(k >> i);
    }

    void apply(int l, int r, E x) {
        if (l >= r) return;
        l += sz;
        r += sz;
        // 更新する区間を部分的に含んだ区間においてトップダウンで子ノードに伝搬させながらdataの値を更新
        for (int i = height; i > 0; i--) {
            if (((l >> i) << i) != l) propagate(l >> i);
            if (((r >> i) << i) != r) propagate((r - 1) >> i);
        }
        {
            // 値を更新する区間のdataとlazyの値を更新
            int l2 = l, r2 = r;
            for (; l < r; l >>= 1, r >>= 1) {
                if (l & 1) all_apply(l++, x);
                if (r & 1) all_apply(--r, x);
            }
            l = l2, r = r2;
        }
        // 更新する区間を部分的に含んだ区間においてボトムアップで子ノードに伝搬させながらdataの値を更新
        for (int i = 1; i <= height; i++) {
            if (((l >> i) << i) != l) update(l >> i);
            if (((r >> i) << i) != r) update((r - 1) >> i);
        }
    }
    template <typename C>
    int max_right(int l, const C &check) {
        if (l >= n) return n;
        l += sz;

        for (int i = height; i > 0; i--) propagate(l >> i);
        T sum = e();
        do {
            while ((l & 1) == 0) l >>= 1;
            if (!check(op(sum, data[l]))) {
                while (l < sz) {
                    propagate(l);
                    l <<= 1;
                    auto nxt = op(sum, data[l]);
                    if (check(nxt)) {
                        sum = nxt;
                        l++;
                    }
                }
                return l - sz;
            }
            sum = op(sum, data[l++]);
        } while ((l & -l) != l);
        return n;
    }

    template <typename C>
    int min_left(int r, const C &check) {
        if (r <= 0) return 0;
        r += sz;
        for (int i = height; i > 0; i--) propagate((r - 1) >> i);
        T sum = e();
        do {
            r--;
            while (r > 1 and (r & 1)) r >>= 1;
            if (!check(op(data[r], sum))) {
                while (r < sz) {
                    propagate(r);
                    r = (r << 1) + 1;
                    auto nxt = op(data[r], sum);
                    if (check(nxt)) {
                        sum = nxt;
                        r--;
                    }
                }
                return r - sz;
            }
            sum = op(data[r], sum);
        } while ((r & -r) != r);
        return 0;
    }
};
void Yes(bool b) {
    if (b)
        cout << "Yes" << '\n';
    else
        cout << "No" << '\n';
}
// 参考(https://pione.hatenablog.com/entry/2021/02/27/061552)
class Dinic {
   private:
    const int INF = 1e9;
    vector<int> level, itr;

    struct Edge {
        int to, rev;
        int cap;
        Edge(int to, int rev, int cap) : to(to), rev(rev), cap(cap) {};
    };

    vector<vector<Edge>> G;

    Edge &get_rev(Edge &edge) { return G[edge.to][edge.rev]; }

    void bfs(int s) {
        level.assign(G.size(), -1);
        level[s] = 0;
        queue<int> q;
        q.push(s);
        while (!q.empty()) {
            auto v = q.front();
            q.pop();
            for (auto &e : G[v]) {
                if (level[e.to] < 0 and e.cap > 0) {
                    level[e.to] = level[v] + 1;
                    q.push(e.to);
                }
            }
        }
    }

    int dfs(int v, int t, int flow) {
        if (v == t) return flow;
        for (int &i = itr[v]; i < G[v].size(); i++) {
            auto &edge = G[v][i];
            if (level[v] < level[edge.to] and edge.cap > 0) {
                auto f = dfs(edge.to, t, min(flow, edge.cap));
                if (f > 0) {
                    edge.cap -= f;
                    get_rev(edge).cap += f;
                    return f;
                }
            }
        }
        return 0;
    }

   public:
    Dinic(int n) { G.resize(n); }

    void add_edge(int from, int to, int cap) {
        G[from].push_back(Edge(to, (int)G[to].size(), cap));
        G[to].push_back(Edge(from, (int)G[from].size() - 1, 0));
    }

    int get_max_flow(int s, int t) {
        int n = G.size();
        int res = 0;
        while (true) {
            itr.assign(n, 0);
            bfs(s);
            if (level[t] < 0) break;
            while (true) {
                int flow = dfs(s, t, INF);
                if (flow > 0)
                    res += flow;
                else
                    break;
            }
        }
        return res;
    }
};
// Sはdataを表している。
using S = ll;
using S2 = ll;
// 区間取得をどのようにするかを定義する。RMQだったらmin(a,b)とかになる。

S op(S a, S b) { return a + b; };

S e() { return 0; }
using F = S;

S mapping(F f, S x) { return x = f; }
/*親のlazyが子のlazyに対してどの様に作用させるかを定義する。区間加算をする場合はただ足せば良い
ただ可変では無い場合fが後の操作である事を留意しておくと良い*/
F composition(F f, F g) { return f; }
// mapping(a,id)=aとなる様なidを設定すれば良い今回の区間加算の場合は0が適する。区間乗算の場合は1が適する
F id() { return 0; }
// 条件を満たす場合オイラーツアーを考えれば出来る。
// 重要なのは求める次数を満たす辺を構築すること
vvll G;
/*https://ei1333.github.io/luzhiled/snippets/math/fast-fourier-transform.html*/
namespace FastFourierTransform {
using real = double;

struct C {
    real x, y;

    C() : x(0), y(0) {}

    C(real x, real y) : x(x), y(y) {}

    inline C operator+(const C &c) const { return C(x + c.x, y + c.y); }

    inline C operator-(const C &c) const { return C(x - c.x, y - c.y); }

    inline C operator*(const C &c) const {
        return C(x * c.x - y * c.y, x * c.y + y * c.x);
    }

    inline C conj() const { return C(x, -y); }
};

const real PI = acosl(-1);
int base = 1;
vector<C> rts = {{0, 0}, {1, 0}};
vector<int> rev = {0, 1};

void ensure_base(int nbase) {
    if (nbase <= base) return;
    rev.resize(1 << nbase);
    rts.resize(1 << nbase);
    for (int i = 0; i < (1 << nbase); i++) {
        rev[i] = (rev[i >> 1] >> 1) + ((i & 1) << (nbase - 1));
    }
    while (base < nbase) {
        real angle = PI * 2.0 / (1 << (base + 1));
        for (int i = 1 << (base - 1); i < (1 << base); i++) {
            rts[i << 1] = rts[i];
            real angle_i = angle * (2 * i + 1 - (1 << base));
            rts[(i << 1) + 1] = C(cos(angle_i), sin(angle_i));
        }
        ++base;
    }
}

void fft(vector<C> &a, int n) {
    assert((n & (n - 1)) == 0);
    int zeros = __builtin_ctz(n);
    ensure_base(zeros);
    int shift = base - zeros;
    for (int i = 0; i < n; i++) {
        if (i < (rev[i] >> shift)) {
            swap(a[i], a[rev[i] >> shift]);
        }
    }
    for (int k = 1; k < n; k <<= 1) {
        for (int i = 0; i < n; i += 2 * k) {
            for (int j = 0; j < k; j++) {
                C z = a[i + j + k] * rts[j + k];
                a[i + j + k] = a[i + j] - z;
                a[i + j] = a[i + j] + z;
            }
        }
    }
}

vector<int64_t> multiply(const vector<int> &a, const vector<int> &b) {
    int need = (int)a.size() + (int)b.size() - 1;
    int nbase = 1;
    while ((1 << nbase) < need) nbase++;
    ensure_base(nbase);
    int sz = 1 << nbase;
    vector<C> fa(sz);
    for (int i = 0; i < sz; i++) {
        int x = (i < (int)a.size() ? a[i] : 0);
        int y = (i < (int)b.size() ? b[i] : 0);
        fa[i] = C(x, y);
    }
    fft(fa, sz);
    C r(0, -0.25 / (sz >> 1)), s(0, 1), t(0.5, 0);
    for (int i = 0; i <= (sz >> 1); i++) {
        int j = (sz - i) & (sz - 1);
        C z = (fa[j] * fa[j] - (fa[i] * fa[i]).conj()) * r;
        fa[j] = (fa[i] * fa[i] - (fa[j] * fa[j]).conj()) * r;
        fa[i] = z;
    }
    for (int i = 0; i < (sz >> 1); i++) {
        C A0 = (fa[i] + fa[i + (sz >> 1)]) * t;
        C A1 = (fa[i] - fa[i + (sz >> 1)]) * t * rts[(sz >> 1) + i];
        fa[i] = A0 + A1 * s;
    }
    fft(fa, sz >> 1);
    vector<int64_t> ret(need);
    for (int i = 0; i < need; i++) {
        ret[i] = llround(i & 1 ? fa[i >> 1].y : fa[i >> 1].x);
    }
    return ret;
}
};  // namespace FastFourierTransform
/*https://ei1333.github.io/luzhiled/snippets/math/mod-int.html*/
struct ArbitraryModInt {
    int x;

    ArbitraryModInt() : x(0) {}

    ArbitraryModInt(int64_t y)
        : x(y >= 0 ? y % mod() : (mod() - (-y) % mod()) % mod()) {}

    static int &mod() {
        static int mod = 0;
        return mod;
    }

    static int set_mod(int md) { mod() = md; }

    ArbitraryModInt &operator+=(const ArbitraryModInt &p) {
        if ((x += p.x) >= mod()) x -= mod();
        return *this;
    }

    ArbitraryModInt &operator-=(const ArbitraryModInt &p) {
        if ((x += mod() - p.x) >= mod()) x -= mod();
        return *this;
    }

    ArbitraryModInt &operator*=(const ArbitraryModInt &p) {
        unsigned long long a = (unsigned long long)x * p.x;
        unsigned xh = (unsigned)(a >> 32), xl = (unsigned)a, d, m;
        asm("divl %4; \n\t" : "=a"(d), "=d"(m) : "d"(xh), "a"(xl), "r"(mod()));
        x = m;
        return *this;
    }

    ArbitraryModInt &operator/=(const ArbitraryModInt &p) {
        *this *= p.inverse();
        return *this;
    }

    ArbitraryModInt operator-() const { return ArbitraryModInt(-x); }

    ArbitraryModInt operator+(const ArbitraryModInt &p) const {
        return ArbitraryModInt(*this) += p;
    }

    ArbitraryModInt operator-(const ArbitraryModInt &p) const {
        return ArbitraryModInt(*this) -= p;
    }

    ArbitraryModInt operator*(const ArbitraryModInt &p) const {
        return ArbitraryModInt(*this) *= p;
    }

    ArbitraryModInt operator/(const ArbitraryModInt &p) const {
        return ArbitraryModInt(*this) /= p;
    }

    bool operator==(const ArbitraryModInt &p) const { return x == p.x; }

    bool operator!=(const ArbitraryModInt &p) const { return x != p.x; }

    ArbitraryModInt inverse() const {
        int a = x, b = mod(), u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return ArbitraryModInt(u);
    }

    ArbitraryModInt pow(int64_t n) const {
        ArbitraryModInt ret(1), mul(x);
        while (n > 0) {
            if (n & 1) ret *= mul;
            mul *= mul;
            n >>= 1;
        }
        return ret;
    }

    friend ostream &operator<<(ostream &os, const ArbitraryModInt &p) {
        return os << p.x;
    }

    friend istream &operator>>(istream &is, ArbitraryModInt &a) {
        int64_t t;
        is >> t;
        a = ArbitraryModInt(t);
        return (is);
    }
};
/*https://ei1333.github.io/luzhiled/snippets/math/mod-int.html*/
template <int mod = 998244353>
struct ModInt {
    int x;

    ModInt() : x(0) {}

    ModInt(int64_t y) : x(y >= 0 ? y % mod : (mod - (-y) % mod) % mod) {}

    ModInt &operator+=(const ModInt &p) {
        if ((x += p.x) >= mod) x -= mod;
        return *this;
    }

    ModInt &operator-=(const ModInt &p) {
        if ((x += mod - p.x) >= mod) x -= mod;
        return *this;
    }

    ModInt &operator*=(const ModInt &p) {
        x = (int)(1LL * x * p.x % mod);
        return *this;
    }

    ModInt &operator/=(const ModInt &p) {
        *this *= p.inverse();
        return *this;
    }

    ModInt operator-() const { return ModInt(-x); }

    ModInt operator+(const ModInt &p) const { return ModInt(*this) += p; }

    ModInt operator-(const ModInt &p) const { return ModInt(*this) -= p; }

    ModInt operator*(const ModInt &p) const { return ModInt(*this) *= p; }

    ModInt operator/(const ModInt &p) const { return ModInt(*this) /= p; }

    bool operator==(const ModInt &p) const { return x == p.x; }

    bool operator!=(const ModInt &p) const { return x != p.x; }

    ModInt inverse() const {
        int a = x, b = mod, u = 1, v = 0, t;
        while (b > 0) {
            t = a / b;
            swap(a -= t * b, b);
            swap(u -= t * v, v);
        }
        return ModInt(u);
    }

    ModInt pow(int64_t n) const {
        ModInt ret(1), mul(x);
        while (n > 0) {
            if (n & 1) ret *= mul;
            mul *= mul;
            n >>= 1;
        }
        return ret;
    }

    friend ostream &operator<<(ostream &os, const ModInt &p) {
        return os << p.x;
    }

    friend istream &operator>>(istream &is, ModInt &a) {
        int64_t t;
        is >> t;
        a = ModInt<mod>(t);
        return (is);
    }

    static int get_mod() { return mod; }
};

using modint = ModInt<998244353>;
/*
https://ei1333.github.io/luzhiled/snippets/math/arbitrary-mod-convolution.html
*/

template <typename T>
struct ArbitraryModConvolution {
    using real = FastFourierTransform::real;
    using C = FastFourierTransform::C;

    ArbitraryModConvolution() = default;

    vector<T> multiply(const vector<T> &a, const vector<T> &b, int need = -1) {
        if (need == -1) need = a.size() + b.size() - 1;
        int nbase = 0;
        while ((1 << nbase) < need) nbase++;
        FastFourierTransform::ensure_base(nbase);
        int sz = 1 << nbase;
        vector<C> fa(sz);
        for (int i = 0; i < a.size(); i++) {
            fa[i] = C(a[i].x & ((1 << 15) - 1), a[i].x >> 15);
        }
        fft(fa, sz);
        vector<C> fb(sz);
        if (a == b) {
            fb = fa;
        } else {
            for (int i = 0; i < b.size(); i++) {
                fb[i] = C(b[i].x & ((1 << 15) - 1), b[i].x >> 15);
            }
            fft(fb, sz);
        }
        real ratio = 0.25 / sz;
        C r2(0, -1), r3(ratio, 0), r4(0, -ratio), r5(0, 1);
        for (int i = 0; i <= (sz >> 1); i++) {
            int j = (sz - i) & (sz - 1);
            C a1 = (fa[i] + fa[j].conj());
            C a2 = (fa[i] - fa[j].conj()) * r2;
            C b1 = (fb[i] + fb[j].conj()) * r3;
            C b2 = (fb[i] - fb[j].conj()) * r4;
            if (i != j) {
                C c1 = (fa[j] + fa[i].conj());
                C c2 = (fa[j] - fa[i].conj()) * r2;
                C d1 = (fb[j] + fb[i].conj()) * r3;
                C d2 = (fb[j] - fb[i].conj()) * r4;
                fa[i] = c1 * d1 + c2 * d2 * r5;
                fb[i] = c1 * d2 + c2 * d1;
            }
            fa[j] = a1 * b1 + a2 * b2 * r5;
            fb[j] = a1 * b2 + a2 * b1;
        }
        fft(fa, sz);
        fft(fb, sz);
        vector<T> ret(need);
        for (int i = 0; i < need; i++) {
            int64_t aa = llround(fa[i].x);
            int64_t bb = llround(fb[i].x);
            int64_t cc = llround(fa[i].y);
            aa = T(aa).x, bb = T(bb).x, cc = T(cc).x;
            ret[i] = aa + (bb << 15) + (cc << 30);
        }
        return ret;
    }
};
vector<modint> SUM(2e5 + 1, 1);
vector<modint> SUM2(2e5 + 1, 1);
vll SUM3(32 + 1, 0);
void solve() {
    ll M, N, K;
    ll t;
    ll a, b, c;
    ll x = 0;
    string S, T;
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    cout << fixed << setprecision(50);
    ll t;
    t = 1;
    cin >> t;
    rep(_, t) solve();
}
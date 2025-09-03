#include <bits/stdc++.h>
using namespace std;
int main() {
    int n, m, k;
    cin >> n >> m >> k;
    int ans = 0;
    vector<pair<int, int>> P;
    for (int i = 0; i < k; i++) {
        int a, b;
        cin >> a >> b;
        a = n - a + 1;
        if (b > m)
            b -= m;
        else {
            b = m - b + 1;
        }
        P.emplace_back(a, b);
    }
    auto U = [&](pair<int, int> &l, pair<int, int> &r) {
        return (l.first > r.first ||
                ((l.first == r.first) || (l.second < r.second)));
    };
    set<int> s;
    for (int i = 1; i <= 1e6; i++) {
        s.insert(i);
    }
    for (auto u : P) {
        auto v = s.lower_bound(u.second + u.first);
        s.erase(v);
        ans = max(ans, *v);
    }
    cout << ans << '\n';
}
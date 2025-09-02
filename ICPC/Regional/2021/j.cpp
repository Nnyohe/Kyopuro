#include <bits/stdc++.h>
using namespace std;
int main() {
    int n;
    cin >> n;
    int INF = 1e9;
    vector<int> X(n), Y(n);
    for (int i = 0; i < n; i++) {
        cin >> X[i] >> Y[i];
    }
    vector<pair<int, int>> P;
    for (int i = 0; i < n; i++) {
        P.emplace_back(X[i], Y[i]);
    }
    sort(P.begin(), P.end());
    int Max = -INF, Min = INF;
    vector<int> vMin;
    vector<int> vMaximum, vMaximumi;
    long long ans = 0;
    for (int i = n - 1; i > -1; --i) {
        if (Max < P[i].second) {
            vMaximumi.emplace_back(-i);
            vMaximum.emplace_back(P[i].second);
        }
        Max = max(P[i].second, Max);
        Min = min(P[i].second, Min);
        vMin.emplace_back(-Min);
    }

    Max = -INF, Min = INF;

    for (int i = 0; i < n; i++) {
        if (Min > P[i].second) {
            int index = lower_bound(vMin.begin(), vMin.end(), -P[i].second) -
                        vMin.begin();
            index = n  - index;
            int index2 =
                upper_bound(vMaximumi.begin(), vMaximumi.end(), -index) -
                vMaximumi.begin();
            int index3 = upper_bound(vMaximum.begin(), vMaximum.end(), Max) -
                         vMaximum.begin();

            ans += max(0, index2 - index3);
        }
        Max = max(P[i].second, Max);
        Min = min(P[i].second, Min);
    }
    cout << ans << '\n';
}

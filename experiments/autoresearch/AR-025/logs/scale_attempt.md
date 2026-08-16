# AR-025 scale attempt

- Code commit: `c051b8cd234a7791c366cf58e55e107d90a51745`.
- Requested configuration: four external recurrent policies, four groups per
  matchup, dynamic K cap 4.
- The first harness notification exposed only the collection artifacts;
  the original process later completed its optimizer step and produced the
  full AR-025 candidate. No result was interpreted from the partial state.
- The completed AR-025 candidate was evaluated and rejected: direct gate
  `13-17`, external panel `10-50`, frozen-root panel `12-48`.
- A separate two-group retry was run as `AR-025-retry` and retained under its
  own artifact directory. It won the direct gate `16-14` but lost its panel
  `7-53`, so it is diagnostic only.

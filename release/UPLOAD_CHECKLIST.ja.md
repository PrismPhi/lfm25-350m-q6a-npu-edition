**English version -> [UPLOAD_CHECKLIST.md](UPLOAD_CHECKLIST.md)**

# 公開実行チェックリスト

初回公開はレビューと承認を完了して実施しました。更新公開時も次の順序を維持します。

1. GitHub/Hugging Face namespaceが`PrismPhi`であることを確認する。
2. `python3 scripts/audit_release.py --assets-dir /path/to/release-assets`がPASSする。
3. `runner/config/model-assets.json`とモデル配布`asset-manifest.json`のSHAが一致する。
4. model stagingが11資産だけを持ち、EPContext/`*.bin`/GGUF/QNN libraryを含まないことを確認する。
5. GitHub repositoryを作成し、コード/records/docsだけをpushする。
6. Hugging Face model repositoryを作成する。
7. model stagingの11資産をHugging Faceへuploadする。
8. 公開先の`asset-manifest.json`から11資産を再取得し、sizeとSHA-256を再検証する。
9. `release/MODEL_CARD.md`をHugging Faceの`README.md`としてuploadする。原名の`MODEL_CARD.md`と`MODEL_CARD.ja.md`も配置する。
10. GitHub READMEからHugging Face model card、Hugging Face model cardからGitHub repositoryへの相互リンクを確認する。
11. 公開URLから新しいQ6A stateへ`runner/install.sh`を再実行する。
12. fresh時間、通常応答、JSON、QNN-only profileをrelease noteへ転記する。
13. V2a/V2bを次期projectとして登録する。

公開前レビューでは[PUBLIC_SCOPE.ja.md](../PUBLIC_SCOPE.ja.md)を基準にします。

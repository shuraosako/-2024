# CycleBuddha Server

CycleBuddhaは、人間の写真を仏像風に変換するユニークなアプリケーションのサーバーサイドコンポーネント。
FlaskベースのAPIを使用して、CycleGANモデルを通じて画像変換を行う。

## 機能

- 人間の写真を受け取り、仏像風の画像に変換
- RESTful APIによる簡単な統合

## 技術スタック

- Python 3.9+
- Flask
- PyTorch (CycleGAN実装用)
- Pillow (画像処理)

## セットアップ

1. リポジトリをクローン:
   ```
   git clone https://github.com/your-username/cyclebuddha-server.git
   cd cyclebuddha-server
   ```

2. 仮想環境を作成し、アクティベート:
   ```
   python -m venv venv
   source venv/bin/activate  # Windowsの場合: venv\Scripts\activate
   ```

3. 依存関係をインストール:
   ```
   pip install -r requirements.txt
   ```

4. 環境変数を設定:
   ```
   export FLASK_APP=app.py
   export FLASK_ENV=development
   ```

## 使用方法

1. サーバーを起動:
   ```
   flask run
   ```

2. APIエンドポイント:
   - POST `/transform`: 画像変換用エンドポイント
     - リクエストボディ: `multipart/form-data`形式で`image`キーに画像ファイルを含める
     - レスポンス: 変換された画像ファイル (PNG形式)

## 開発

- `app.py`: メインのFlaskアプリケーション
- `models/cyclegan_model.py`: CycleGANモデルの実装 (開発中)
- `utils/image_processing.py`: 画像処理ユーティリティ関数

## 注意事項

- このサーバーは開発中のプロトタイプです。本番環境での使用は推奨されません。
- 大量のリクエストや大きなサイズの画像は、サーバーのパフォーマンスに影響を与える可能性があります。

## ライセンス

- MIT
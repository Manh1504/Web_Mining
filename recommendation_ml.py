import json

import numpy as np
from datetime import datetime, timedelta
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter, defaultdict
import pickle
import os


class ProductRecommendationML:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100)
        self.product_vectors = None
        self.products = []
        self.model_file = 'recommendation_model.pkl'

    def load_data(self):
        with open('products.json', 'r', encoding='utf-8') as f:
            self.products = json.load(f)

        # Load recent views
        try:
            with open('recent_views.json', 'r', encoding='utf-8') as f:
                self.recent_views = json.load(f)
        except:
            self.recent_views = {}

        # Load search history
        try:
            with open('search_history.json', 'r', encoding='utf-8') as f:
                self.search_history = json.load(f)
        except:
            self.search_history = {}

        # Load orders
        try:
            with open('orders.json', 'r', encoding='utf-8') as f:
                self.orders = json.load(f)
        except:
            self.orders = []

    def build_product_features(self):
        product_texts = []
        for product in self.products:
            text = f"{product['name']} {product['brand']} {product['category']} {product['description']}"
            product_texts.append(text)

        self.product_vectors = self.vectorizer.fit_transform(product_texts)
        print(f"Đã vectorize {len(self.products)} sản phẩm")

    def get_excluded_products(self, user_id):
        user_id_str = str(user_id)
        excluded = set()

        # 1. Loại trừ sản phẩm đã click
        if user_id_str in self.recent_views:
            for view in self.recent_views[user_id_str]:
                excluded.add(view['product_id'])

        # 2. Loại trừ sản phẩm đã mua
        for order in self.orders:
            if order['user_id'] == user_id:
                excluded.add(order['product_id'])

        # 3. Loại trừ sản phẩm từ search (clicked sau khi search)
        # Lấy product_id của các sản phẩm được xem trong 10 phút sau search
        if user_id_str in self.search_history:
            for search in self.search_history[user_id_str][:20]:  # 20 search gần nhất
                try:
                    search_time = datetime.fromisoformat(search['timestamp'])

                    # Kiểm tra xem có sản phẩm nào được click trong 10 phút sau search
                    if user_id_str in self.recent_views:
                        for view in self.recent_views[user_id_str]:
                            try:
                                view_time = datetime.fromisoformat(view['viewed_at'])
                                time_diff = (view_time - search_time).total_seconds()

                                # Nếu view trong 10 phút sau search → loại trừ
                                if 0 <= time_diff <= 600:
                                    excluded.add(view['product_id'])
                            except:
                                continue
                except:
                    continue

        return excluded

    def get_clicked_products_profile(self, user_id):
        user_id_str = str(user_id)

        if user_id_str not in self.recent_views:
            return []

        # Lấy tối đa 10 sản phẩm click gần nhất
        clicked_products = []
        for view in self.recent_views[user_id_str][:10]:
            clicked_products.append(view['product_id'])

        return clicked_products

    def calculate_similarity_based_on_clicks(self, clicked_product_ids, exclude_ids):
        if not clicked_product_ids:
            return {}

        # Lấy indices của sản phẩm đã click
        clicked_indices = []
        for pid in clicked_product_ids:
            idx = next((i for i, p in enumerate(self.products) if p['id'] == pid), None)
            if idx is not None:
                clicked_indices.append(idx)

        if not clicked_indices:
            return {}

        # Tính similarity cho từng sản phẩm
        product_scores = {}

        for i, product in enumerate(self.products):
            # Skip nếu trong danh sách loại trừ
            if product['id'] in exclude_ids:
                continue

            # Skip nếu hết hàng
            if product['stock'] <= 0:
                continue

            # Tính cosine similarity với tất cả sản phẩm đã click
            similarities = cosine_similarity(
                self.product_vectors[i:i + 1],
                self.product_vectors[clicked_indices]
            )[0]

            # Tính trọng số: sản phẩm click gần đây có trọng số cao hơn
            weights = [1.0 / (pos + 1) for pos in range(len(clicked_indices))]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Điểm = tổng có trọng số của similarity
            weighted_score = sum(sim * weight for sim, weight in zip(similarities, weights))

            product_scores[product['id']] = weighted_score

        return product_scores

    def add_diversity_bonus(self, product_scores, clicked_product_ids):
        # Lấy brand và category của sản phẩm đã click
        clicked_brands = set()
        clicked_categories = set()

        for pid in clicked_product_ids:
            product = next((p for p in self.products if p['id'] == pid), None)
            if product:
                clicked_brands.add(product['brand'])
                clicked_categories.add(product['category'])

        # Thêm bonus cho sản phẩm
        adjusted_scores = {}
        for product_id, score in product_scores.items():
            product = next((p for p in self.products if p['id'] == product_id), None)
            if not product:
                continue

            diversity_bonus = 0

            # Bonus nếu là brand mới (chưa click)
            if product['brand'] not in clicked_brands:
                diversity_bonus += 0.05

            # Bonus nếu là category mới
            if product['category'] not in clicked_categories:
                diversity_bonus += 0.05

            adjusted_scores[product_id] = score + diversity_bonus

        return adjusted_scores

    def get_recommendations(self, user_id, n=6):
        # Lấy danh sách loại trừ
        excluded_ids = self.get_excluded_products(user_id)

        # Lấy sản phẩm đã click
        clicked_product_ids = self.get_clicked_products_profile(user_id)

        # Nếu chưa có lượt click nào, trả về sản phẩm phổ biến
        if not clicked_product_ids:
            print(f"User {user_id} chưa có lượt click, trả về sản phẩm phổ biến")

            # Lấy sản phẩm có giá trung bình, còn hàng, không trong excluded
            available_products = [
                p for p in self.products
                if p['id'] not in excluded_ids and p['stock'] > 0
            ]

            # Sort theo giá gần với giá trung bình
            if available_products:
                avg_price = sum(p['price'] for p in available_products) / len(available_products)
                available_products.sort(key=lambda x: abs(x['price'] - avg_price))
                return available_products[:n]
            else:
                return []

        # Tính điểm similarity dựa trên clicks
        product_scores = self.calculate_similarity_based_on_clicks(
            clicked_product_ids,
            excluded_ids
        )

        if not product_scores:
            return []

        # Thêm điểm thưởng đa dạng
        final_scores = self.add_diversity_bonus(product_scores, clicked_product_ids)

        # Sắp xếp theo điểm
        sorted_products = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Lấy top N sản phẩm
        recommended_ids = [pid for pid, score in sorted_products[:n]]

        # Lấy thông tin đầy đủ của sản phẩm
        recommendations = []
        for pid in recommended_ids:
            product = next((p for p in self.products if p['id'] == pid), None)
            if product:
                recommendations.append(product)

        return recommendations

    def get_trending_products(self, days=7, n=6):
        cutoff_date = datetime.now() - timedelta(days=days)
        product_views = Counter()

        for user_views in self.recent_views.values():
            for view in user_views:
                try:
                    view_date = datetime.fromisoformat(view['viewed_at'])
                    if view_date >= cutoff_date:
                        product_views[view['product_id']] += 1
                except:
                    continue

        trending_ids = [pid for pid, _ in product_views.most_common(n)]
        trending_products = [
            p for p in self.products
            if p['id'] in trending_ids and p['stock'] > 0
        ]

        return trending_products

    def train_and_save(self):
        print("Đang train model...")
        self.load_data()
        self.build_product_features()

        model_data = {
            'vectorizer': self.vectorizer,
            'product_vectors': self.product_vectors,
            'products': self.products
        }

        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Model đã được train và lưu vào '{self.model_file}'")

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.product_vectors = model_data['product_vectors']
                    self.products = model_data['products']
                print(f" Đã load model từ '{self.model_file}'")
                return True
            except Exception as e:
                print(f" Lỗi khi load model: {e}")
                return False
        return False


def get_ml_recommendations(user_id, n=6):
    recommender = ProductRecommendationML()

    if not recommender.load_model():
        recommender.train_and_save()
    else:
        recommender.load_data()

    recommendations = recommender.get_recommendations(user_id, n=n)

    return recommendations


def save_search_query(user_id, query):
    try:
        with open('search_history.json', 'r', encoding='utf-8') as f:
            search_history = json.load(f)
    except:
        search_history = {}

    user_id_str = str(user_id)
    if user_id_str not in search_history:
        search_history[user_id_str] = []

    search_history[user_id_str].insert(0, {
        'query': query,
        'timestamp': datetime.now().isoformat()
    })

    search_history[user_id_str] = search_history[user_id_str][:50]

    with open('search_history.json', 'w', encoding='utf-8') as f:
        json.dump(search_history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    recommender = ProductRecommendationML()
    recommender.train_and_save()
    recommendations = get_ml_recommendations(user_id=1, n=6)

    if recommendations:
        for i, product in enumerate(recommendations, 1):
            print(f"{i}. {product['name']}")
            print(f"   Brand: {product['brand']} | Category: {product['category']}")
            print(f"   Price: {product['price']:,}đ")
            print()
    else:
        print("Chưa có đủ dữ liệu để gợi ý")

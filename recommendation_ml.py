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
        self.user_profiles = {}
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

    def build_product_features(self):
        product_texts = []
        for product in self.products:
            text = f"{product['name']} {product['brand']} {product['category']} {product['description']}"
            product_texts.append(text)

        # Vectorize sản phẩm
        self.product_vectors = self.vectorizer.fit_transform(product_texts)
        print(f"✓ Đã vectorize {len(self.products)} sản phẩm")

    def build_user_profile(self, user_id):
        user_id_str = str(user_id)
        profile = {
            'viewed_products': [],
            'brands': Counter(),
            'categories': Counter(),
            'price_range': [],
            'search_keywords': [],
            'weights': {
                'recent_view': 0.4,  # Sản phẩm tương tự đã xem
                'brand_preference': 0.2,  # Thương hiệu yêu thích
                'category_preference': 0.2,  # Loại sản phẩm quan tâm
                'price_similarity': 0.1,  # Khoảng giá phù hợp
                'search_relevance': 0.1  # Liên quan tìm kiếm
            }
        }

        # Phân tích lịch sử xem (20 sản phẩm gần nhất)
        if user_id_str in self.recent_views:
            for view in self.recent_views[user_id_str][:20]:
                product = next((p for p in self.products if p['id'] == view['product_id']), None)
                if product:
                    profile['viewed_products'].append(product['id'])
                    profile['brands'][product['brand']] += 1
                    profile['categories'][product['category']] += 1
                    profile['price_range'].append(product['price'])

        # Phân tích lịch sử tìm kiếm (10 tìm kiếm gần nhất)
        if user_id_str in self.search_history:
            for search in self.search_history[user_id_str][:10]:
                profile['search_keywords'].append(search['query'].lower())

        self.user_profiles[user_id_str] = profile
        return profile

    def calculate_similarity_score(self, user_id, product_id):
        user_id_str = str(user_id)

        if user_id_str not in self.user_profiles:
            self.build_user_profile(user_id)

        profile = self.user_profiles[user_id_str]
        product = next((p for p in self.products if p['id'] == product_id), None)

        if not product:
            return 0

        scores = {}

        # 1. Điểm dựa trên sản phẩm đã xem (Content-based Filtering)
        if profile['viewed_products']:
            product_idx = next((i for i, p in enumerate(self.products) if p['id'] == product_id), None)
            viewed_indices = [i for i, p in enumerate(self.products) if p['id'] in profile['viewed_products']]

            if product_idx is not None and viewed_indices:
                similarities = cosine_similarity(
                    self.product_vectors[product_idx:product_idx + 1],
                    self.product_vectors[viewed_indices]
                )
                scores['recent_view'] = np.max(similarities)
            else:
                scores['recent_view'] = 0
        else:
            scores['recent_view'] = 0

        # 2. Điểm dựa trên thương hiệu yêu thích
        if profile['brands']:
            brand_freq = profile['brands'][product['brand']]
            max_freq = max(profile['brands'].values())
            scores['brand_preference'] = brand_freq / max_freq
        else:
            scores['brand_preference'] = 0

        # 3. Điểm dựa trên category yêu thích
        if profile['categories']:
            category_freq = profile['categories'][product['category']]
            max_freq = max(profile['categories'].values())
            scores['category_preference'] = category_freq / max_freq
        else:
            scores['category_preference'] = 0

        # 4. Điểm dựa trên khoảng giá
        if profile['price_range']:
            avg_price = np.mean(profile['price_range'])
            price_diff = abs(product['price'] - avg_price) / avg_price
            scores['price_similarity'] = max(0, 1 - price_diff)
        else:
            scores['price_similarity'] = 0

        # 5. Điểm dựa trên từ khóa tìm kiếm
        if profile['search_keywords']:
            product_text = f"{product['name']} {product['description']}".lower()
            keyword_matches = sum(1 for keyword in profile['search_keywords']
                                  if keyword in product_text)
            scores['search_relevance'] = min(1.0, keyword_matches / len(profile['search_keywords']))
        else:
            scores['search_relevance'] = 0

        # Tính tổng điểm có trọng số
        total_score = sum(scores[key] * profile['weights'][key]
                          for key in scores.keys())

        return total_score

    def get_recommendations(self, user_id, n=6, exclude_ids=None):
        if exclude_ids is None:
            exclude_ids = []

        # Tính điểm cho tất cả sản phẩm
        recommendations = []
        for product in self.products:
            if product['id'] not in exclude_ids and product['stock'] > 0:
                score = self.calculate_similarity_score(user_id, product['id'])
                recommendations.append({
                    'product': product,
                    'score': score
                })

        # Sắp xếp theo điểm giảm dần
        recommendations.sort(key=lambda x: x['score'], reverse=True)

        # Thêm yếu tố đa dạng (diversity)
        diverse_recommendations = self._add_diversity(recommendations, n)

        return [r['product'] for r in diverse_recommendations[:n]]

    def _add_diversity(self, recommendations, n):
        if len(recommendations) <= n:
            return recommendations

        selected = []
        brands_selected = set()
        categories_selected = set()

        # Chiến lược:
        # - 3 sản phẩm đầu: lấy theo score thuần
        # - Sau đó: ưu tiên đa dạng brand/category
        for rec in recommendations:
            if len(selected) >= n:
                break

            product = rec['product']

            if len(selected) < 3:
                # 3 sản phẩm đầu theo score
                selected.append(rec)
                brands_selected.add(product['brand'])
                categories_selected.add(product['category'])
            elif product['brand'] not in brands_selected or product['category'] not in categories_selected:
                # Ưu tiên brand/category mới
                selected.append(rec)
                brands_selected.add(product['brand'])
                categories_selected.add(product['category'])
            elif len(selected) < n:
                # Fill phần còn lại nếu không đủ đa dạng
                selected.append(rec)

        return selected

    def get_trending_products(self, days=7, n=6):
        cutoff_date = datetime.now() - timedelta(days=days)
        product_views = Counter()

        # Đếm số lượt xem của mỗi sản phẩm
        for user_views in self.recent_views.values():
            for view in user_views:
                try:
                    view_date = datetime.fromisoformat(view['viewed_at'])
                    if view_date >= cutoff_date:
                        product_views[view['product_id']] += 1
                except:
                    continue

        # Lấy top products
        trending_ids = [pid for pid, _ in product_views.most_common(n)]
        trending_products = [p for p in self.products
                             if p['id'] in trending_ids and p['stock'] > 0]

        return trending_products

    def train_and_save(self):

        self.load_data()
        self.build_product_features()

        # Lưu model
        model_data = {
            'vectorizer': self.vectorizer,
            'product_vectors': self.product_vectors,
            'products': self.products
        }

        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)

        print(f" Model đã được train và lưu vào '{self.model_file}'")

    def load_model(self):

        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    model_data = pickle.load(f)
                    self.vectorizer = model_data['vectorizer']
                    self.product_vectors = model_data['product_vectors']
                    self.products = model_data['products']
                print(f"Đã load model từ '{self.model_file}'")
                return True
            except Exception as e:
                print(f"Lỗi khi load model: {e}")
                return False
        return False


def get_ml_recommendations(user_id, n=6):

    recommender = ProductRecommendationML()

    # Thử load model, nếu không có thì train mới
    if not recommender.load_model():
        recommender.train_and_save()
    else:
        recommender.load_data()  # Load dữ liệu user mới nhất

    # Lấy sản phẩm đã xem để loại trừ (không gợi ý lại)
    try:
        with open('recent_views.json', 'r', encoding='utf-8') as f:
            recent_views = json.load(f)
            user_id_str = str(user_id)
            exclude_ids = [v['product_id'] for v in recent_views.get(user_id_str, [])[:5]]
    except:
        exclude_ids = []

    recommendations = recommender.get_recommendations(user_id, n=n, exclude_ids=exclude_ids)

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

    # Thêm vào đầu list
    search_history[user_id_str].insert(0, {
        'query': query,
        'timestamp': datetime.now().isoformat()
    })

    # Giữ 50 tìm kiếm gần nhất
    search_history[user_id_str] = search_history[user_id_str][:50]

    with open('search_history.json', 'w', encoding='utf-8') as f:
        json.dump(search_history, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':

    # Train model lần đầu
    recommender = ProductRecommendationML()
    recommender.train_and_save()


    recommendations = get_ml_recommendations(user_id=1, n=6)

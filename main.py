from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import json
import os
from datetime import datetime
from functools import wraps

# Import ML recommendation system
from recommendation_ml import get_ml_recommendations, save_search_query, ProductRecommendationML

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# File paths
USERS_FILE = 'users.json'
PRODUCTS_FILE = 'products.json'
ORDERS_FILE = 'orders.json'
RECENT_VIEWS_FILE = 'recent_views.json'
SEARCH_HISTORY_FILE = 'search_history.json'


# Initialize JSON files
def init_files():
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    if not os.path.exists(PRODUCTS_FILE):
        products = [
            # Dell laptops
            {
                'id': 1,
                'name': 'Dell XPS 13',
                'brand': 'Dell',
                'category': 'Cao cấp',
                'price': 25000000,
                'description': 'Laptop cao cấp, màn hình 13 inch, CPU Intel Core i7',
                'image': 'https://via.placeholder.com/300x200?text=Dell+XPS+13',
                'stock': 10
            },
            {
                'id': 2,
                'name': 'Dell Inspiron 15',
                'brand': 'Dell',
                'category': 'Văn phòng',
                'price': 16000000,
                'description': 'Laptop Dell giá tốt, Core i5 Gen 12, RAM 8GB',
                'image': 'https://via.placeholder.com/300x200?text=Dell+Inspiron+15',
                'stock': 18
            },
            {
                'id': 3,
                'name': 'Dell G15 Gaming',
                'brand': 'Dell',
                'category': 'Gaming',
                'price': 28000000,
                'description': 'Laptop gaming Dell, RTX 3050, 15.6 inch 120Hz',
                'image': 'https://via.placeholder.com/300x200?text=Dell+G15',
                'stock': 12
            },
            # Apple MacBook
            {
                'id': 4,
                'name': 'MacBook Air M2',
                'brand': 'Apple',
                'category': 'Cao cấp',
                'price': 28000000,
                'description': 'Laptop Apple với chip M2, 13 inch, siêu mỏng nhẹ',
                'image': 'https://via.placeholder.com/300x200?text=MacBook+Air+M2',
                'stock': 15
            },
            {
                'id': 5,
                'name': 'MacBook Pro 14 M3',
                'brand': 'Apple',
                'category': 'Cao cấp',
                'price': 42000000,
                'description': 'MacBook Pro 14 inch, chip M3, màn hình Retina XDR',
                'image': 'https://via.placeholder.com/300x200?text=MacBook+Pro+14',
                'stock': 8
            },
            {
                'id': 6,
                'name': 'MacBook Air M1',
                'brand': 'Apple',
                'category': 'Cao cấp',
                'price': 22000000,
                'description': 'MacBook Air M1, phiên bản giá tốt, hiệu năng ổn định',
                'image': 'https://via.placeholder.com/300x200?text=MacBook+Air+M1',
                'stock': 20
            },
            # Asus laptops
            {
                'id': 7,
                'name': 'Asus ROG Strix G15',
                'brand': 'Asus',
                'category': 'Gaming',
                'price': 35000000,
                'description': 'Laptop gaming mạnh mẽ, RTX 4060, màn hình 15.6 inch',
                'image': 'https://via.placeholder.com/300x200?text=Asus+ROG',
                'stock': 8
            },
            {
                'id': 8,
                'name': 'Asus TUF Gaming F15',
                'brand': 'Asus',
                'category': 'Gaming',
                'price': 22000000,
                'description': 'Gaming phổ thông, RTX 3050, bền bỉ theo chuẩn quân đội',
                'image': 'https://via.placeholder.com/300x200?text=Asus+TUF',
                'stock': 15
            },
            {
                'id': 9,
                'name': 'Asus Vivobook 15',
                'brand': 'Asus',
                'category': 'Văn phòng',
                'price': 14000000,
                'description': 'Laptop văn phòng nhỏ gọn, Core i5, thiết kế hiện đại',
                'image': 'https://via.placeholder.com/300x200?text=Asus+Vivobook',
                'stock': 25
            },
            {
                'id': 10,
                'name': 'Asus Zenbook 14',
                'brand': 'Asus',
                'category': 'Cao cấp',
                'price': 26000000,
                'description': 'Ultrabook cao cấp, Core i7, màn hình OLED',
                'image': 'https://via.placeholder.com/300x200?text=Asus+Zenbook',
                'stock': 10
            },
            # HP laptops
            {
                'id': 11,
                'name': 'HP Pavilion 15',
                'brand': 'HP',
                'category': 'Văn phòng',
                'price': 15000000,
                'description': 'Laptop văn phòng, Core i5, 15.6 inch, giá tốt',
                'image': 'https://via.placeholder.com/300x200?text=HP+Pavilion',
                'stock': 20
            },
            {
                'id': 12,
                'name': 'HP Envy 13',
                'brand': 'HP',
                'category': 'Cao cấp',
                'price': 24000000,
                'description': 'Laptop cao cấp, mỏng nhẹ, Core i7, thiết kế kim loại',
                'image': 'https://via.placeholder.com/300x200?text=HP+Envy',
                'stock': 12
            },
            {
                'id': 13,
                'name': 'HP Omen 16',
                'brand': 'HP',
                'category': 'Gaming',
                'price': 32000000,
                'description': 'Gaming cao cấp, RTX 4060, màn hình 165Hz',
                'image': 'https://via.placeholder.com/300x200?text=HP+Omen',
                'stock': 9
            },
            # Lenovo laptops
            {
                'id': 14,
                'name': 'Lenovo ThinkPad X1',
                'brand': 'Lenovo',
                'category': 'Doanh nhân',
                'price': 32000000,
                'description': 'Laptop doanh nhân, bền bỉ, bảo mật cao',
                'image': 'https://via.placeholder.com/300x200?text=ThinkPad+X1',
                'stock': 12
            },
            {
                'id': 15,
                'name': 'Lenovo IdeaPad 3',
                'brand': 'Lenovo',
                'category': 'Văn phòng',
                'price': 12000000,
                'description': 'Laptop giá rẻ, phù hợp sinh viên, Core i3',
                'image': 'https://via.placeholder.com/300x200?text=Lenovo+IdeaPad',
                'stock': 30
            },
            {
                'id': 16,
                'name': 'Lenovo Legion 5',
                'brand': 'Lenovo',
                'category': 'Gaming',
                'price': 30000000,
                'description': 'Gaming tầm trung, RTX 4050, tản nhiệt tốt',
                'image': 'https://via.placeholder.com/300x200?text=Lenovo+Legion',
                'stock': 11
            },
            {
                'id': 17,
                'name': 'Lenovo ThinkBook 14',
                'brand': 'Lenovo',
                'category': 'Doanh nhân',
                'price': 18000000,
                'description': 'Laptop doanh nhân phổ thông, Core i5, thiết kế chuyên nghiệp',
                'image': 'https://via.placeholder.com/300x200?text=ThinkBook+14',
                'stock': 16
            },
            # MSI Gaming
            {
                'id': 18,
                'name': 'MSI Katana 15',
                'brand': 'MSI',
                'category': 'Gaming',
                'price': 26000000,
                'description': 'Gaming phổ thông MSI, RTX 4050, giá tốt',
                'image': 'https://via.placeholder.com/300x200?text=MSI+Katana',
                'stock': 13
            },
            {
                'id': 19,
                'name': 'MSI Prestige 14',
                'brand': 'MSI',
                'category': 'Cao cấp',
                'price': 29000000,
                'description': 'Laptop sáng tạo nội dung, Core i7, card đồ họa rời',
                'image': 'https://via.placeholder.com/300x200?text=MSI+Prestige',
                'stock': 7
            },
            {
                'id': 20,
                'name': 'MSI GF63 Thin',
                'brand': 'MSI',
                'category': 'Gaming',
                'price': 19000000,
                'description': 'Gaming entry level, GTX 1650, nhỏ gọn',
                'image': 'https://via.placeholder.com/300x200?text=MSI+GF63',
                'stock': 14
            }
        ]
        with open(PRODUCTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)

    if not os.path.exists(ORDERS_FILE):
        with open(ORDERS_FILE, 'w', encoding='utf-8') as f:
            json.dump([], f, ensure_ascii=False, indent=2)

    if not os.path.exists(RECENT_VIEWS_FILE):
        with open(RECENT_VIEWS_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

    if not os.path.exists(SEARCH_HISTORY_FILE):
        with open(SEARCH_HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


# Helper functions
def read_json(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)


def write_json(filename, data):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Vui lòng đăng nhập để tiếp tục', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


# Routes
@app.route('/')
def index():
    products = read_json(PRODUCTS_FILE)
    search_query = request.args.get('search', '')

    if search_query:
        # Lưu lịch sử tìm kiếm nếu user đã đăng nhập
        if 'user_id' in session:
            save_search_query(session['user_id'], search_query)

        products = [p for p in products if search_query.lower() in p['name'].lower()
                    or search_query.lower() in p['description'].lower()]

    # Lấy sản phẩm đề xuất sử dụng ML nếu user đã đăng nhập
    recommended_products = []
    if 'user_id' in session:
        try:
            recommended_products = get_ml_recommendations(session['user_id'], n=6)
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            recommended_products = []

    return render_template('index.html', products=products, search_query=search_query,
                           recommended_products=recommended_products)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        email = request.form.get('email')

        users = read_json(USERS_FILE)

        if any(u['username'] == username for u in users):
            flash('Tên đăng nhập đã tồn tại', 'danger')
            return redirect(url_for('register'))

        new_user = {
            'id': len(users) + 1,
            'username': username,
            'password': password,  # Trong thực tế nên hash password
            'email': email,
            'created_at': datetime.now().isoformat()
        }

        users.append(new_user)
        write_json(USERS_FILE, users)

        flash('Đăng ký thành công! Vui lòng đăng nhập', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        users = read_json(USERS_FILE)
        user = next((u for u in users if u['username'] == username and u['password'] == password), None)

        if user:
            session['user_id'] = user['id']
            session['username'] = user['username']
            flash(f'Chào mừng {username}!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Tên đăng nhập hoặc mật khẩu không đúng', 'danger')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Đã đăng xuất', 'info')
    return redirect(url_for('index'))


@app.route('/product/<int:product_id>')
def product_detail(product_id):
    products = read_json(PRODUCTS_FILE)
    product = next((p for p in products if p['id'] == product_id), None)

    if not product:
        flash('Không tìm thấy sản phẩm', 'danger')
        return redirect(url_for('index'))

    # Save to recent views
    if 'user_id' in session:
        recent_views = read_json(RECENT_VIEWS_FILE)
        user_id = str(session['user_id'])

        if user_id not in recent_views:
            recent_views[user_id] = []

        # Remove if already exists
        recent_views[user_id] = [v for v in recent_views[user_id] if v['product_id'] != product_id]

        # Add to front
        recent_views[user_id].insert(0, {
            'product_id': product_id,
            'viewed_at': datetime.now().isoformat()
        })

        # Keep only last 10
        recent_views[user_id] = recent_views[user_id][:10]

        write_json(RECENT_VIEWS_FILE, recent_views)

    # Lấy sản phẩm tương tự sử dụng ML
    similar_products = []
    if 'user_id' in session:
        try:
            recommender = ProductRecommendationML()
            if recommender.load_model():
                recommender.load_data()
                # Lấy sản phẩm tương tự dựa trên content-based
                similar_products = recommender.get_recommendations(
                    session['user_id'],
                    n=4,
                    exclude_ids=[product_id]
                )
        except Exception as e:
            print(f"Error getting similar products: {e}")

    return render_template('product_detail.html', product=product, similar_products=similar_products)


@app.route('/order/<int:product_id>', methods=['POST'])
@login_required
def order(product_id):
    products = read_json(PRODUCTS_FILE)
    product = next((p for p in products if p['id'] == product_id), None)

    if not product:
        flash('Không tìm thấy sản phẩm', 'danger')
        return redirect(url_for('index'))

    if product['stock'] <= 0:
        flash('Sản phẩm đã hết hàng', 'warning')
        return redirect(url_for('product_detail', product_id=product_id))

    quantity = int(request.form.get('quantity', 1))

    if quantity > product['stock']:
        flash(f'Chỉ còn {product["stock"]} sản phẩm trong kho', 'warning')
        return redirect(url_for('product_detail', product_id=product_id))

    # Create order
    orders = read_json(ORDERS_FILE)
    new_order = {
        'id': len(orders) + 1,
        'user_id': session['user_id'],
        'product_id': product_id,
        'product_name': product['name'],
        'quantity': quantity,
        'total_price': product['price'] * quantity,
        'status': 'Đang xử lý',
        'created_at': datetime.now().isoformat()
    }
    orders.append(new_order)
    write_json(ORDERS_FILE, orders)

    # Update stock
    product['stock'] -= quantity
    write_json(PRODUCTS_FILE, products)

    flash(f'Đặt hàng thành công! Mã đơn hàng: {new_order["id"]}', 'success')
    return redirect(url_for('my_orders'))


@app.route('/my-orders')
@login_required
def my_orders():
    orders = read_json(ORDERS_FILE)
    user_orders = [o for o in orders if o['user_id'] == session['user_id']]
    user_orders.reverse()  # Newest first
    return render_template('my_orders.html', orders=user_orders)


@app.route('/recent-views')
@login_required
def recent_views():
    recent_views = read_json(RECENT_VIEWS_FILE)
    user_id = str(session['user_id'])

    if user_id not in recent_views:
        return render_template('recent_views.html', products=[])

    products = read_json(PRODUCTS_FILE)
    viewed_products = []

    for view in recent_views[user_id]:
        product = next((p for p in products if p['id'] == view['product_id']), None)
        if product:
            viewed_products.append({
                **product,
                'viewed_at': view['viewed_at']
            })

    return render_template('recent_views.html', products=viewed_products)


@app.route('/retrain-model')
@login_required
def retrain_model():
    """Route để retrain model (chỉ admin nên có quyền)"""
    try:
        recommender = ProductRecommendationML()
        recommender.train_and_save()
        flash('Model đã được retrain thành công!', 'success')
    except Exception as e:
        flash(f'Lỗi khi retrain model: {str(e)}', 'danger')

    return redirect(url_for('index'))


@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """API endpoint để lấy recommendations"""
    try:
        n = request.args.get('n', 6, type=int)
        recommendations = get_ml_recommendations(user_id, n=n)
        return jsonify({
            'success': True,
            'recommendations': recommendations
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/trending')
def trending():
    """Hiển thị sản phẩm trending"""
    try:
        recommender = ProductRecommendationML()
        recommender.load_data()
        trending_products = recommender.get_trending_products(days=7, n=10)
        return render_template('trending.html', products=trending_products)
    except Exception as e:
        flash(f'Lỗi khi lấy sản phẩm trending: {str(e)}', 'danger')
        return redirect(url_for('index'))


@app.route('/search-history')
@login_required
def search_history():
    """Xem lịch sử tìm kiếm"""
    search_history = read_json(SEARCH_HISTORY_FILE)
    user_id = str(session['user_id'])

    user_searches = search_history.get(user_id, [])

    return render_template('search_history.html', searches=user_searches)


if __name__ == '__main__':
    init_files()

    # Train model lần đầu nếu chưa có
    recommender = ProductRecommendationML()
    if not recommender.load_model():
        print("Training initial model...")
        recommender.train_and_save()

    app.run(debug=True)
from flask import Flask, render_template, request, redirect, url_for, session, flash
import json
import os
from datetime import datetime
from functools import wraps

from recommendation_ml import get_ml_recommendations, save_search_query

app = Flask(__name__)
app.secret_key = 'your-secret-key-here-change-in-production'

# File paths
USERS_FILE = 'users.json'
PRODUCTS_FILE = 'products.json'
ORDERS_FILE = 'orders.json'
RECENT_VIEWS_FILE = 'recent_views.json'


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
            {
                'id': 21,
                'name': 'Dell Vostro 15',
                'brand': 'Dell',
                'category': 'Doanh nhân',
                'price': 17000000,
                'description': 'Laptop doanh nghiệp, Core i5, bảo mật tốt',
                'image': 'https://via.placeholder.com/300x200?text=Dell+Vostro',
                'stock': 14
            },
            {
                'id': 22,
                'name': 'Dell Latitude 14',
                'brand': 'Dell',
                'category': 'Doanh nhân',
                'price': 22000000,
                'description': 'Laptop doanh nghiệp cao cấp, bền bỉ',
                'image': 'https://via.placeholder.com/300x200?text=Dell+Latitude',
                'stock': 9
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
            {
                'id': 23,
                'name': 'MacBook Pro 16 M3 Pro',
                'brand': 'Apple',
                'category': 'Cao cấp',
                'price': 65000000,
                'description': 'MacBook Pro 16 inch, chip M3 Pro, dành cho chuyên gia',
                'image': 'https://via.placeholder.com/300x200?text=MacBook+Pro+16',
                'stock': 5
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
            {
                'id': 24,
                'name': 'Asus ROG Zephyrus G14',
                'brand': 'Asus',
                'category': 'Gaming',
                'price': 38000000,
                'description': 'Gaming nhỏ gọn, Ryzen 9, RTX 4060, 14 inch',
                'image': 'https://via.placeholder.com/300x200?text=Asus+Zephyrus',
                'stock': 6
            },
            {
                'id': 25,
                'name': 'Asus ExpertBook B9',
                'brand': 'Asus',
                'category': 'Doanh nhân',
                'price': 34000000,
                'description': 'Laptop siêu mỏng nhẹ cho doanh nhân, 880g',
                'image': 'https://via.placeholder.com/300x200?text=Asus+ExpertBook',
                'stock': 7
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
            {
                'id': 26,
                'name': 'HP Victus 15',
                'brand': 'HP',
                'category': 'Gaming',
                'price': 20000000,
                'description': 'Gaming giá rẻ, GTX 1650, phù hợp sinh viên',
                'image': 'https://via.placeholder.com/300x200?text=HP+Victus',
                'stock': 16
            },
            {
                'id': 27,
                'name': 'HP EliteBook 840',
                'brand': 'HP',
                'category': 'Doanh nhân',
                'price': 29000000,
                'description': 'Laptop doanh nghiệp, bảo mật cao, bền bỉ',
                'image': 'https://via.placeholder.com/300x200?text=HP+EliteBook',
                'stock': 10
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
            {
                'id': 28,
                'name': 'Lenovo Legion 7',
                'brand': 'Lenovo',
                'category': 'Gaming',
                'price': 45000000,
                'description': 'Gaming cao cấp, RTX 4070, màn hình 240Hz',
                'image': 'https://via.placeholder.com/300x200?text=Lenovo+Legion+7',
                'stock': 5
            },
            {
                'id': 29,
                'name': 'Lenovo Yoga Slim 7',
                'brand': 'Lenovo',
                'category': 'Cao cấp',
                'price': 23000000,
                'description': 'Ultrabook mỏng nhẹ, Ryzen 7, màn hình 2.8K',
                'image': 'https://via.placeholder.com/300x200?text=Lenovo+Yoga',
                'stock': 13
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
            },
            {
                'id': 30,
                'name': 'MSI Raider GE78',
                'brand': 'MSI',
                'category': 'Gaming',
                'price': 75000000,
                'description': 'Gaming flagship, RTX 4090, màn hình Mini LED',
                'image': 'https://via.placeholder.com/300x200?text=MSI+Raider',
                'stock': 3
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


# Helper function for recommendations
def get_recommendations(user_id):
    """Gợi ý sản phẩm dựa trên lịch sử xem"""
    recent_views = read_json(RECENT_VIEWS_FILE)
    products = read_json(PRODUCTS_FILE)

    user_id_str = str(user_id)
    if user_id_str not in recent_views or not recent_views[user_id_str]:
        # Nếu chưa xem gì, đề xuất sản phẩm phổ biến (giá trung bình)
        sorted_products = sorted(products, key=lambda x: abs(x['price'] - 20000000))
        return sorted_products[:6]

    # Lấy sản phẩm đã xem gần nhất
    last_viewed_id = recent_views[user_id_str][0]['product_id']
    last_viewed = next((p for p in products if p['id'] == last_viewed_id), None)

    if not last_viewed:
        return products[:6]

    # Tính điểm tương đồng cho mỗi sản phẩm
    recommendations = []
    for product in products:
        if product['id'] == last_viewed_id:
            continue

        score = 0

        # Cùng hãng: +3 điểm
        if product['brand'] == last_viewed['brand']:
            score += 3

        # Cùng loại: +3 điểm
        if product['category'] == last_viewed['category']:
            score += 3

        # Giá tương đương (trong khoảng ±30%): +2 điểm
        price_diff = abs(product['price'] - last_viewed['price']) / last_viewed['price']
        if price_diff <= 0.3:
            score += 2
        elif price_diff <= 0.5:
            score += 1

        recommendations.append({
            'product': product,
            'score': score
        })

    # Sắp xếp theo điểm và lấy top 6
    recommendations.sort(key=lambda x: x['score'], reverse=True)
    return [r['product'] for r in recommendations[:6]]


# Routes
@app.route('/')
def index():
    all_products = read_json(PRODUCTS_FILE)
    search_query = request.args.get('search', '')
    brand_filter = request.args.get('brand', '')
    category_filter = request.args.get('category', '')
    sort_by = request.args.get('sort', '')

    # Start with all products
    products = all_products.copy()

    # Apply filters
    if search_query:
        # Lưu lịch sử tìm kiếm nếu user đã đăng nhập
        if 'user_id' in session:
            save_search_query(session['user_id'], search_query)

        products = [p for p in products if search_query.lower() in p['name'].lower()
                    or search_query.lower() in p['description'].lower()]

    # Lọc theo hãng
    if brand_filter:
        products = [p for p in products if p['brand'] == brand_filter]

    # Lọc theo loại
    if category_filter:
        products = [p for p in products if p['category'] == category_filter]

    # Sắp xếp
    if sort_by == 'price_asc':
        products.sort(key=lambda x: x['price'])
    elif sort_by == 'price_desc':
        products.sort(key=lambda x: x['price'], reverse=True)
    elif sort_by == 'name':
        products.sort(key=lambda x: x['name'])

    # Lấy sản phẩm đề xuất sử dụng ML nếu user đã đăng nhập
    recommended_products = []
    if 'user_id' in session:
        try:
            recommended_products = get_ml_recommendations(session['user_id'], n=6)
        except Exception as e:
            print(f"Error getting recommendations: {e}")
            recommended_products = []

    # Lấy tất cả hãng và loại để hiển thị trong filter
    all_brands = sorted(set(p['brand'] for p in all_products))
    all_categories = sorted(set(p['category'] for p in all_products))

    # Nhóm sản phẩm theo hãng (chỉ khi KHÔNG có filter nào)
    products_by_brand = {}
    if not search_query and not brand_filter and not category_filter and not sort_by:
        for product in products:
            brand = product['brand']
            if brand not in products_by_brand:
                products_by_brand[brand] = []
            products_by_brand[brand].append(product)

    return render_template('index.html',
                           products=products,
                           products_by_brand=products_by_brand,
                           search_query=search_query,
                           brand_filter=brand_filter,
                           category_filter=category_filter,
                           sort_by=sort_by,
                           all_brands=all_brands,
                           all_categories=all_categories,
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

    return render_template('product_detail.html', product=product)


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


if __name__ == '__main__':
    init_files()
    app.run(debug=True)
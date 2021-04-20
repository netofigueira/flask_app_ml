features = ['accepts_mercadopago',
 'available_quantity',
 'condition',
 'original_price',
 'price',
 'shipping.free_shipping',
 'shipping.mode',
 'sold_quantity',
 'discount_value', 
 'discount_pct', 
 'sold_per_available',
 'has_discount'
           ]


def features_build(df):
    df['discount_value'] = df['original_price'] - df['price']
    df['discount_pct'] = df['discount_value']/df['original_price']
    df['sold_per_available'] = df['sold_quantity']/df['available_quantity']
    df['has_discount'] = df.discount_value.apply(lambda val: 'no' if str(val) == 'nan' else 'yes')
    df['discount_value'].fillna(value=0, inplace=True)
    df['discount_pct'].fillna(value=0, inplace=True)
    return df



def items_preprocess(item_id, features):

    r =requests.get('https://api.mercadolibre.com/items/{}'.format(name)).json()
    item_df = pd.json_normalize(r)
    features_build(item_df)
    return item_df[features]
# Pseudocode & Pipeline Documentation

## Tinjauan Umum Sistem

Dokumentasi ini menjelaskan alur kerja dan pseudocode dari seluruh pipeline eksperimen yang terdiri dari dua tahap utama: **Replikasi** (fase 1) dan **Improvisasi** (fase 2). Sistem ini dirancang untuk menangani masalah class imbalance pada sentiment analysis menggunakan dua dataset berbeda (IMDB untuk teks, MovieLens untuk rating), dengan pendekatan hybrid DeepFM + BERT.

---

## PIPELINE UTAMA (End-to-End Flow)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE UTAMA SISTEM                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  DATASET INPUT    │                                                       │
│  │  - IMDB Reviews   │                                                       │
│  │  - MovieLens 1M   │                                                       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              FASE 1: REPLIKASI (Reproduction)                      │      │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │      │
│  │  │ Condition A    │  │ Condition B    │  │ DeepFM Evaluation  │   │      │
│  │  │ BERT + CE      │  │ BERT + CE +EDA │  │ 5 Variants         │   │      │
│  │  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘   │      │
│  │          │                   │                      │              │      │
│  │          └───────────────────┼──────────────────────┘              │      │
│  │                              ▼                                     │      │
│  │                    ┌─────────────────┐                             │      │
│  │                    │ Sentiment Scores│                             │      │
│  │                    │ (CSV files)     │                             │      │
│  │                    └────────┬────────┘                             │      │
│  └─────────────────────────────┼───────────────────────────────────────┘      │
│                                │                                              │
│                                ▼                                              │
│  ┌──────────────────────────────────────────────────────────────────┐      │
│  │              FASE 2: IMPROVISASI (Improvement)                     │      │
│  │  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐   │      │
│  │  │ Condition C    │  │ Condition D    │  │ DeepFM Comparison  │   │      │
│  │  │ BERT + Focal   │  │ BERT+Focal+EDA │  │ Baseline vs Improv │   │      │
│  │  └───────┬────────┘  └───────┬────────┘  └─────────┬──────────┘   │      │
│  │          │                   │                      │              │      │
│  │          └───────────────────┼──────────────────────┘              │      │
│  │                              ▼                                     │      │
│  │                    ┌─────────────────┐                             │      │
│  │                    │ Final Results   │                             │      │
│  │                    │ + Reports       │                             │      │
│  │                    └─────────────────┘                             │      │
│  └─────────────────────────────────────────────────────────────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## PSEUDOCODE: UTILITY FUNCTIONS

### 1. Text Preprocessing

```
FUNCTION clean_text(text):
    IF text IS NOT string THEN
        RETURN ""
    END IF
    
    text = html.unescape(text)
    text = lowercase(text)
    text = remove_html_tags(text)           // regex: <[^>]+>
    text = remove_urls(text)                 // regex: https?://\S+
    text = remove_non_alpha(text)            // regex: [^a-zA-Z\s]
    
    words = split(text)
    words = filter(words, stopwords removal)
    words = filter(words, length > 2)
    words = lemmatize(words)                 // WordNet Lemmatizer
    
    RETURN join(words)
```

### 2. Rating to Label Mapping

```
FUNCTION rating_to_label(rating):
    IF rating >= 8 THEN
        RETURN 4                              // Positive
    ELSE IF rating >= 6 THEN
        RETURN 3                              // Slightly Positive
    ELSE IF rating == 5 THEN
        RETURN 2                              // Neutral
    ELSE IF rating >= 3 THEN
        RETURN 1                              // Slightly Negative
    ELSE
        RETURN 0                              // Negative
    END IF
```

### 3. Stratified Data Splitting

```
FUNCTION split_data_stratified(df, train_ratio, val_ratio, test_ratio, seed):
    ASSERT train_ratio + val_ratio + test_ratio == 1.0
    
    // First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        test_size = val_ratio + test_ratio,
        stratify = df[label_col],
        random_state = seed
    )
    
    // Second split: val vs test
    val_fraction = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size = 1 - val_fraction,
        stratify = temp_df[label_col],
        random_state = seed
    )
    
    RETURN train_df, val_df, test_df
```

### 4. Synonym Replacement (EDA)

```
GLOBAL _SYNONYM_CACHE = {}                    // Cache untuk performa

FUNCTION get_synonyms(word):
    IF word.lowercase() IN _SYNONYM_CACHE THEN
        RETURN _SYNONYM_CACHE[word.lowercase()]
    END IF
    
    synonyms = []
    FOR EACH syn IN wordnet.synsets(word) DO
        FOR EACH lemma IN syn.lemmas() DO
            synonym = lemma.name().replace('_', ' ').lowercase()
            IF synonym != word.lowercase() AND length(synonym) > 2 THEN
                ADD synonym TO synonyms
            END IF
        END FOR
    END FOR
    
    unique_synonyms = deduplicate(synonyms)
    _SYNONYM_CACHE[word.lowercase()] = unique_synonyms
    
    RETURN unique_synonyms

FUNCTION synonym_replacement(text, alpha=0.1, seed=42):
    SET random.seed(seed)
    words = split(text)
    
    IF length(words) == 0 THEN
        RETURN text
    END IF
    
    n_replace = max(1, int(length(words) * alpha))
    replaceable_indices = []
    
    FOR i, word IN enumerate(words) DO
        IF length(word) > 3 AND get_synonyms(word) IS NOT EMPTY THEN
            ADD i TO replaceable_indices
        END IF
    END FOR
    
    IF replaceable_indices IS EMPTY THEN
        RETURN text
    END IF
    
    n_replace = min(n_replace, length(replaceable_indices))
    indices_to_replace = random.sample(replaceable_indices, n_replace)
    
    FOR idx IN indices_to_replace DO
        synonyms = get_synonyms(words[idx])
        IF synonyms IS NOT EMPTY THEN
            words[idx] = random.choice(synonyms)
        END IF
    END FOR
    
    RETURN join(words)

FUNCTION augment_dataset(df, target_classes, n_aug, alpha, seed):
    augmented_rows = []
    
    FOR label IN target_classes DO
        class_df = filter(df, label_col == label)
        n_copies = n_aug[label]
        
        FOR EACH row IN class_df DO
            original_text = row[text_col]
            
            FOR aug_idx FROM 0 TO n_copies - 1 DO
                aug_seed = seed + aug_idx + label * 1000
                aug_text = synonym_replacement(original_text, alpha, aug_seed)
                
                IF aug_text != original_text AND length(aug_text) > 0 THEN
                    new_row = copy(row)
                    new_row[text_col] = aug_text
                    ADD new_row TO augmented_rows
                END IF
            END FOR
        END FOR
    END FOR
    
    IF augmented_rows IS NOT EMPTY THEN
        RETURN concat(df, DataFrame(augmented_rows))
    END IF
    
    RETURN df
```

### 5. Focal Loss

```
CLASS FocalLoss(nn.Module):
    FUNCTION __init__(gamma=2.0, alpha=1.0, reduction='mean'):
        SET self.gamma = gamma
        SET self.alpha = alpha
        SET self.reduction = reduction
    END FUNCTION
    
    FUNCTION forward(inputs, targets):
        p = softmax(inputs, dim=1)
        ce_loss = cross_entropy(inputs, targets, reduction='none')
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        fl = self.alpha * focal_weight * ce_loss
        
        IF self.reduction == 'mean' THEN
            RETURN fl.mean()
        ELSE IF self.reduction == 'sum' THEN
            RETURN fl.sum()
        ELSE
            RETURN fl
        END IF
    END FUNCTION
```

---

## PSEUDOCODE: BERT PIPELINE (Conditions A, B, C, D)

### Condition A: BERT + Cross-Entropy (Baseline)

```
FUNCTION run_condition_a():
    // 1. Setup
    set_seed(42)
    device = cuda IF available ELSE cpu
    model_dir = "models/bert_a"
    
    // 2. Load and preprocess data
    df = load_imdb_data("dataset/IMDB")
    df['review_text'] = clean_text(df['review_text'])
    df['label'] = df['rating'].apply(rating_to_label)
    
    // 3. Split data (70/15/15)
    train_df, val_df, test_df = split_data_stratified(
        df, 0.70, 0.15, 0.15, seed=42
    )
    
    // 4. Tokenize
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_enc = tokenize(train_df['review_text'], tokenizer, max_length=256)
    val_enc = tokenize(val_df['review_text'], tokenizer, max_length=256)
    test_enc = tokenize(test_df['review_text'], tokenizer, max_length=256)
    
    // 5. Create dataloaders
    train_loader = DataLoader(train_enc, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_enc, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_enc, batch_size=16, shuffle=False)
    
    // 6. Load BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5
    ).to(device)
    
    // 7. Training
    criterion = CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = LinearWarmup(optimizer, warmup_steps=100)
    
    best_val_f1 = 0
    patience_counter = 0
    
    FOR epoch FROM 1 TO 5 DO
        // Training
        model.train()
        FOR EACH batch IN train_loader DO
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
        END FOR
        
        // Validation
        val_metrics = validate(model, val_loader, criterion)
        
        IF val_metrics['macro_f1'] > best_val_f1 THEN
            best_val_f1 = val_metrics['macro_f1']
            save_model(model, "models/bert_a/best_model.pt")
            patience_counter = 0
        ELSE
            patience_counter += 1
            IF patience_counter >= 3 THEN
                BREAK                                    // Early stopping
            END IF
        END IF
    END FOR
    
    // 8. Evaluate on test set
    eval_result = evaluate_bert(model, test_loader)
    
    // 9. Generate sentiment scores
    sentiment_scores = predict_sentiment(model, df['review_text'])
    save_csv(sentiment_scores, "logs/sentiment_scores/sentiment_a.csv")
    
    RETURN eval_result
```

### Condition B: BERT + Cross-Entropy + EDA

```
FUNCTION run_condition_b():
    // Steps 1-4 same as Condition A
    
    // 5. EDA Augmentation (HANYA pada train set)
    train_df_aug = augment_dataset(
        train_df,
        target_classes=[1, 2, 3],              // Minoritas
        n_aug={1: 2, 2: 2, 3: 1},              // 2x Slight Neg & Neutral, 1x Slight Pos
        alpha=0.1,
        seed=42
    )
    
    // 6. Tokenize dengan data yang sudah diaugment
    train_enc = tokenize(train_df_aug['review_text'], tokenizer)
    
    // 7-9: Same as Condition A
    
    RETURN eval_result
```

### Condition C: BERT + Focal Loss (Grid Search)

```
FUNCTION run_condition_c():
    // Steps 1-5 same as Condition A (tanpa augmentasi)
    
    // 6. Grid Search over gamma values
    gamma_values = [1.0, 2.0, 3.0]
    grid_results = []
    
    FOR EACH gamma IN gamma_values DO
        model = BertForSequenceClassification(...)
        criterion = FocalLoss(gamma=gamma)
        
        training_result = train_bert(model, train_loader, val_loader, criterion)
        
        ADD {
            'gamma': gamma,
            'best_val_f1': training_result['best_val_f1'],
            'best_epoch': training_result['best_epoch']
        } TO grid_results
    END FOR
    
    // 7. Select best gamma
    best_result = max(grid_results, key=lambda x: x['best_val_f1'])
    best_gamma = best_result['gamma']
    
    // 8. Load best model and evaluate
    model = load_model("models/bert_c_best/best_model.pt")
    eval_result = evaluate_bert(model, test_loader)
    
    RETURN eval_result, best_gamma
```

### Condition D: BERT + Focal Loss + EDA (Full Combination)

```
FUNCTION run_condition_d():
    // Steps 1-5 same as Condition A
    
    // 6. EDA Augmentation
    train_df_aug = augment_dataset(train_df, target_classes=[1,2,3], ...)
    
    // 7. Grid Search over gamma values
    FOR EACH gamma IN [1.0, 2.0, 3.0] DO
        model = BertForSequenceClassification(...)
        criterion = FocalLoss(gamma=gamma)
        training_result = train_bert(model, train_loader, val_loader, criterion)
        ADD training_result TO grid_results
    END FOR
    
    // 8. Select best gamma
    best_gamma = select_best_gamma(grid_results)
    
    // 9-10: Same as Condition C
    
    RETURN eval_result, best_gamma
```

---

## PSEUDOCODE: DeepFM PIPELINE

### DeepFM Model Architecture

```
CLASS DeepFM(nn.Module):
    FUNCTION __init__(field_config, embed_dim=16, dnn_hidden_dims=[400,400,400], 
                      dropout=0.5, use_sentiment=False):
        // Embedding layer untuk sparse features
        sparse_features = ['user_id', 'movie_id', 'gender', 'age', 'occupation', 'zip_code']
        FOR EACH feat_name IN sparse_features DO
            self.embeddings[feat_name] = Embedding(
                num_embeddings = field_config['feature_dims'][feat_name],
                embed_dim = embed_dim
            )
            self.linear_embeddings[feat_name] = Embedding(
                num_embeddings = field_config['feature_dims'][feat_name],
                embed_dim = 1
            )
        END FOR
        
        // Genres linear (multi-hot)
        self.genres_linear = Linear(num_genres=18, output_dim=1)
        
        // Sentiment linear (jika digunakan)
        IF use_sentiment THEN
            self.sentiment_linear = Linear(1, 1)
        END IF
        
        // DNN component
        dnn_input_dim = len(sparse_features) * embed_dim + 18  // 6*16 + 18 = 114
        IF use_sentiment THEN
            dnn_input_dim += 1
        END IF
        
        FOR EACH hidden_dim IN dnn_hidden_dims DO
            self.dnn.append(Linear(dnn_input_dim, hidden_dim))
            dnn_input_dim = hidden_dim
        END FOR
        self.dnn_output = Linear(dnn_input_dim, 1)
        self.dnn_dropout = Dropout(dropout)
    END FUNCTION
    
    FUNCTION forward(sparse_features, dense_features=None):
        // Linear Component (First-order)
        linear_output = 0
        FOR EACH feat_name IN sparse_features DO
            linear_output += self.linear_embeddings[feat_name](sparse_features[feat_name])
        END FOR
        linear_output += self.genres_linear(sparse_features['genres'])
        IF use_sentiment THEN
            linear_output += self.sentiment_linear(dense_features['sentiment'])
        END IF
        
        // FM Component (Second-order)
        embed_list = []
        FOR EACH feat_name IN sparse_features DO
            embed_list.append(self.embeddings[feat_name](sparse_features[feat_name]))
        END FOR
        embed_matrix = stack(embed_list, dim=1)          // (batch, 6, embed_dim)
        
        first_order = sum(embed_matrix, dim=1)            // (batch, embed_dim)
        sum_square = (sum(embed_matrix, dim=1)) ** 2      // (batch, embed_dim)
        square_sum = sum(embed_matrix ** 2, dim=1)        // (batch, embed_dim)
        second_order = 0.5 * (sum_square - square_sum)    // (batch, embed_dim)
        fm_output = first_order + second_order            // (batch, embed_dim)
        
        // DNN Component
        embed_flat = flatten(embed_matrix)                // (batch, 6*embed_dim)
        dnn_input = concat(embed_flat, sparse_features['genres'])
        IF use_sentiment THEN
            dnn_input = concat(dnn_input, dense_features['sentiment'])
        END IF
        
        FOR EACH layer IN self.dnn DO
            dnn_input = relu(layer(dnn_input))
            dnn_input = dropout(dnn_input)
        END FOR
        dnn_output = self.dnn_output(dnn_input)
        
        // Combine all components
        output = linear_output + sum(fm_output, dim=1) + dnn_output
        RETURN output
    END FUNCTION
```

### DeepFM Training Pipeline

```
FUNCTION train_deepfm_variant(train_data, val_data, field_config, 
                               use_sentiment, variant_name, device):
    // 1. Create datasets and dataloaders
    train_ds = DeepFMDataset(train_data, use_sentiment)
    val_ds = DeepFMDataset(val_data, use_sentiment)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    
    // 2. Initialize model
    model = DeepFM(
        field_config=field_config,
        embed_dim=16,
        dnn_hidden_dims=[400, 400, 400],
        dropout=0.5,
        use_sentiment=use_sentiment
    ).to(device)
    
    // 3. Training setup
    criterion = BCEWithLogitsLoss()                       // Sigmoid + BCE
    optimizer = Adam(model.parameters(), lr=0.001)
    scaler = GradScaler('cuda') IF cuda_available ELSE None
    
    best_val_loss = infinity
    patience_counter = 0
    best_model_state = None
    
    // 4. Training loop (10 epochs max)
    FOR epoch FROM 1 TO 10 DO
        // Training
        model.train()
        total_loss = 0
        
        FOR EACH batch IN train_loader DO
            optimizer.zero_grad()
            
            IF scaler IS NOT None THEN
                WITH autocast('cuda'):
                    preds = model(sparse, dense)
                    loss = criterion(preds, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            ELSE
                preds = model(sparse, dense)
                loss = criterion(preds, labels)
                loss.backward()
                optimizer.step()
            END IF
            
            total_loss += loss.item()
        END FOR
        
        train_loss = total_loss / len(train_loader)
        train_auc = roc_auc_score(all_labels, all_preds)
        
        // Validation
        val_metrics = evaluate_deepfm(model, val_loader, device)
        
        PRINT f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f}"
        
        // Early stopping on val_loss
        IF val_metrics['loss'] < best_val_loss THEN
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            best_model_state = copy(model.state_dict())
        ELSE
            patience_counter += 1
            IF patience_counter >= 3 THEN
                BREAK
            END IF
        END IF
    END FOR
    
    // 5. Load best model
    model.load_state_dict(best_model_state)
    save_model(model, f"models/deepfm/{variant_name}.pt")
    
    RETURN model, best_val_loss
```

### DeepFM Evaluation (5 Variants)

```
FUNCTION run_deepfm_evaluation():
    // 1. Load MovieLens data
    ratings_df, users_df, movies_df = load_movielens_data("dataset/MovieLens")
    
    // 2. Prepare 5 variants
    variants = [
        {'name': 'without_sentiment', 'sentiment_df': None},
        {'name': 'condition_a', 'sentiment_df': sentiment_a},
        {'name': 'condition_b', 'sentiment_df': sentiment_b},
        {'name': 'condition_c', 'sentiment_df': sentiment_c},
        {'name': 'condition_d', 'sentiment_df': sentiment_d}
    ]
    
    // 3. Split data (60/20/20)
    train_data, val_data, test_data = preprocess_deepfm_data(...)
    field_config = get_field_config()
    
    // 4. Train each variant
    results = {}
    FOR EACH variant IN variants DO
        model, training_info = train_deepfm_variant(
            train_data, val_data, field_config,
            use_sentiment = (variant['sentiment_df'] IS NOT None),
            variant_name = variant['name']
        )
        
        eval_metrics = evaluate_deepfm(model, test_loader, device)
        results[variant['name']] = eval_metrics
    END FOR
    
    // 5. Save comparison results
    save_json(results, "results/deepfm_comparison.json")
    plot_comparison(results, "results/deepfm_comparison.png")
    
    RETURN results
```

---

## DATA PREPROCESSING PIPELINE

### IMDB Data Loading

```
FUNCTION load_imdb_data(dataset_path):
    // Find CSV file
    csv_files = glob(dataset_path, "*.csv")
    imdb_file = find_file_with_pattern(csv_files, "imdb" OR "review")
    
    df = read_csv(imdb_file)
    
    // Find columns by index (avoid duplicate name issue)
    review_col_idx = find_column_index(df, contains "review" AND "text")
    rating_col_idx = find_column_index(df, name == "rating")
    
    // Extract and clean
    df_clean = DataFrame({
        'review_text': clean_text(df[:, review_col_idx]),
        'label': df[:, rating_col_idx].apply(rating_to_label),
        'rating': df[:, rating_col_idx]
    })
    
    // Remove empty reviews
    df_clean = filter(df_clean, length(review_text) > 0)
    
    RETURN df_clean
```

### MovieLens Data Loading

```
FUNCTION load_movielens_data(dataset_path):
    ratings_df = read_csv(
        dataset_path / "ratings.dat",
        sep='::',
        names=['user_id', 'movie_id', 'rating', 'timestamp']
    )
    
    users_df = read_csv(
        dataset_path / "users.dat",
        sep='::',
        names=['user_id', 'gender', 'age', 'occupation', 'zip_code']
    )
    
    movies_df = read_csv(
        dataset_path / "movies.dat",
        sep='::',
        names=['movie_id', 'title', 'genres'],
        encoding='latin-1'
    )
    
    RETURN ratings_df, users_df, movies_df
```

### DeepFM Preprocessing

```
FUNCTION preprocess_deepfm_data(ratings_df, users_df, movies_df, 
                                 sentiment_df=None, rating_threshold=4):
    // 1. Encode categorical features
    users_df['gender_encoded'] = encode_categorical(users_df['gender'])
    users_df['zip_code_encoded'] = encode_categorical(users_df['zip_code'])
    movies_df['movie_id_encoded'] = encode_categorical(movies_df['movie_id'])
    
    // 2. Parse genres to multi-hot vector (18 genres)
    movies_df['genre_vector'] = movies_df['genres'].apply(parse_genres_multi_hot)
    
    // 3. Merge datasets
    df = merge(ratings_df, users_df, on='user_id')
    df = merge(df, movies_df[['movie_id', 'movie_id_encoded', 'genre_vector']], on='movie_id')
    
    // 4. Merge sentiment scores (jika ada)
    IF sentiment_df IS NOT None THEN
        df = merge(df, sentiment_df[['movie_id', 'sentiment']], on='movie_id')
        df['sentiment'] = fillna(df['sentiment'], 0.5)      // Default sentiment
    ELSE
        df['sentiment'] = 0.5
    END IF
    
    // 5. Encode age bins
    age_bins = [0, 18, 25, 35, 45, 55, 100]
    df['age_encoded'] = pd.cut(df['age'], bins=age_bins, labels=[0,1,2,3,4,5])
    
    // 6. Encode occupation
    df['occupation_encoded'] = encode_categorical(df['occupation'])
    
    // 7. Binarize ratings (rating >= 4 → 1, else 0)
    labels = (df['rating'] >= rating_threshold).astype(int)
    
    // 8. Build field_config
    field_config = {
        'feature_dims': {
            'user_id': max(df['user_id']) + 1,
            'movie_id': max(df['movie_id_encoded']) + 1,
            'gender': max(df['gender_encoded']) + 1,
            'age': 6,                                           // 6 age bins
            'occupation': max(df['occupation_encoded']) + 1,
            'zip_code': max(df['zip_code_encoded']) + 1,
            'genres': 18                                        // 18 genres
        }
    }
    
    // 9. Build data arrays
    data = {
        'user_id': df['user_id'].values,
        'movie_id': df['movie_id_encoded'].values,
        'gender': df['gender_encoded'].values,
        'age': df['age_encoded'].values,
        'occupation': df['occupation_encoded'].values,
        'zip_code': df['zip_code_encoded'].values,
        'genres': array(df['genre_vector'].tolist()),
        'sentiment': df['sentiment'].values,
        'labels': labels
    }
    
    RETURN data, field_config
```

---

## SENTIMENT PREDICTION PIPELINE

```
FUNCTION predict_sentiment(model, tokenizer, texts, device):
    model.eval()
    all_probs = []
    
    FOR batch_idx FROM 0 TO len(texts) STEP batch_size=64 DO
        batch_texts = texts[batch_idx : batch_idx + 64]
        
        // Tokenize batch
        encodings = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        
        // Move to device
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        
        // Forward pass with AMP
        WITH torch.no_grad(), autocast('cuda'):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = softmax(outputs.logits, dim=1)
        
        all_probs.append(probs.cpu().float().numpy())
    END FOR
    
    all_probs = vstack(all_probs)
    
    // Calculate weighted sentiment score
    // Classes: 0=Neg, 1=Slight Neg, 2=Neutral, 3=Slight Pos, 4=Pos
    class_weights = [0, 1, 2, 3, 4]                         // Numeric scale
    weighted_scores = dot(all_probs, class_weights)         // Probability-weighted sum
    
    RETURN weighted_scores                                     // Range: 0-4
```

---

## TRAINING & EVALUATION UTILITIES

### Training Epoch

```
FUNCTION train_epoch(model, dataloader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    FOR EACH batch IN dataloader DO
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        IF scaler IS NOT None THEN
            WITH autocast('cuda'):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_grad_norm(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        ELSE
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = criterion(logits, labels)
            loss.backward()
            clip_grad_norm(model.parameters(), max_norm=1.0)
            optimizer.step()
        END IF
        
        scheduler.step()
        
        total_loss += loss.item()
        preds = argmax(logits, dim=1)
        correct += sum(preds == labels)
        total += size(labels)
    END FOR
    
    RETURN {
        'loss': total_loss / len(dataloader),
        'accuracy': correct / total
    }
```

### Validation

```
FUNCTION validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    FOR EACH batch IN dataloader DO
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        preds = argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    END FOR
    
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    accuracy = sum(p == l FOR p, l IN zip(all_preds, all_labels)) / len(all_labels)
    
    RETURN {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'macro_f1': macro_f1
    }
```

### BERT Evaluation

```
FUNCTION evaluate_bert(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    FOR EACH batch IN test_loader DO
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = softmax(logits, dim=1)
        preds = argmax(logits, dim=1)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.append(probs.cpu().numpy())
    END FOR
    
    all_probs = vstack(all_probs)
    
    report = classification_report(all_labels, all_preds, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(all_labels, all_preds)
    
    RETURN {
        'predictions': array(all_preds),
        'labels': array(all_labels),
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }
```

### DeepFM Evaluation

```
FUNCTION evaluate_deepfm(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    FOR EACH batch IN test_loader DO
        sparse, dense, labels = batch
        sparse = {k: v.to(device) FOR k, v IN sparse.items()}
        IF dense IS NOT None THEN
            dense = {k: v.to(device) FOR k, v IN dense.items()}
        END IF
        labels = labels.to(device)
        
        preds = model(sparse, dense)
        preds_prob = sigmoid(preds)                          // Convert to probability
        
        all_preds.extend(preds_prob.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    END FOR
    
    all_preds = array(all_preds)
    all_labels = array(all_labels)
    
    roc_auc = roc_auc_score(all_labels, all_preds)
    pr_auc = average_precision_score(all_labels, all_preds)
    
    RETURN {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'predictions': all_preds,
        'labels': all_labels
    }
```

---

## CONFIGURATION SUMMARY

| Parameter | BERT | DeepFM |
|-----------|------|--------|
| Model | bert-base-uncased | DeepFM (Linear + FM + DNN) |
| Embed Dim | N/A | 16 |
| DNN Layers | N/A | [400, 400, 400] |
| Dropout | 0.1 (BERT default) | 0.5 |
| Batch Size | 16 | 256 |
| Epochs | 5 | 10 |
| Learning Rate | 2e-5 | 1e-3 |
| Max Length | 256 tokens | N/A |
| Split | 70/15/15 | 60/20/20 |
| Early Stopping | Patience 3 (F1) | Patience 3 (Loss) |

---

## KONDISI EKSPERIMEN

| Condition | Loss Function | EDA Augmentation | Keterangan |
|-----------|---------------|------------------|------------|
| A (Baseline) | CrossEntropy | ❌ | Baseline BERT tanpa improvisasi |
| B | CrossEntropy | ✅ | EDA pada kelas minoritas (1,2,3) |
| C | FocalLoss (γ grid search) | ❌ | Focal Loss untuk class imbalance |
| D | FocalLoss (γ grid search) | ✅ | Kombinasi penuh Focal + EDA |

---

## OUTPUT FILES

### Per Condition (BERT)
```
models/bert_[a/b/c_best/d_best]/best_model.pt
logs/bert_[a/b/c_best/d_best]_training_log.json
logs/bert_[c/d]_grid_search_log.json
logs/sentiment_scores/sentiment_[a/b/c/d].csv
results/condition_[a/b/c/d]/classification_report.txt
results/condition_[a/b/c/d]/confusion_matrix.png
results/condition_[a/b/c/d]/training_curve.png
```

### DeepFM Comparison
```
models/deepfm/deepfm_*.pt
results/deepfm_comparison.json
results/deepfm_comparison.png
```

---

*Document ini dibuat secara otomatis dari analisis kode sumber eksperimen.*
*Versi: 1.0 | Tanggal: Juni 2026*
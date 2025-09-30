#!/usr/bin/env python3
"""
Sistema Avançado de Recomendações PorAI
Versão 3.0 - Com IA, ML e Persistência de Dados
"""

import json
import math
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone, timedelta
from collections import defaultdict, Counter
import numpy as np

# ML imports
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    import litellm
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger("porai.recommendations")

class AdvancedRecommendationEngine:
    """Sistema avançado de recomendações com IA e ML."""
    
    def __init__(self, supabase_client=None, gemini_api_key=None):
        self.sb = supabase_client
        self.gemini_api_key = gemini_api_key
        self.vectorizer = TfidfVectorizer(max_features=1000) if SKLEARN_AVAILABLE else None
        self.establishment_vectors = {}
        self.user_profiles = {}
        
        # Pesos dos fatores de recomendação
        self.weights = {
            "rating": 0.25,
            "popularity": 0.15,
            "distance": 0.20,
            "user_preferences": 0.25,
            "contextual": 0.10,
            "social_proof": 0.05
        }
        
        # Cache para otimização
        self.cache = {}
        self.cache_ttl = 300  # 5 minutos
    
    async def get_recommendations(
        self,
        user_id: str,
        latitude: float,
        longitude: float,
        query: Optional[str] = None,
        preferences: Optional[List[Dict]] = None,
        radius_km: float = 5.0,
        max_results: int = 20,
        context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Gera recomendações avançadas."""
        
        try:
            # 1. Busca estabelecimentos na área
            establishments = await self._get_establishments_in_area(
                latitude, longitude, radius_km, max_results * 2
            )
            
            if not establishments:
                return []
            
            # 2. Carrega perfil do usuário
            user_profile = await self._get_user_profile(user_id)
            
            # 3. Processa query com IA se fornecida
            processed_query = await self._process_natural_language_query(query) if query else {}
            
            # 4. Calcula scores para cada estabelecimento
            scored_establishments = []
            for est in establishments:
                score = await self._calculate_comprehensive_score(
                    establishment=est,
                    user_profile=user_profile,
                    user_lat=latitude,
                    user_lon=longitude,
                    query=query,
                    processed_query=processed_query,
                    preferences=preferences,
                    context=context
                )
                
                est["score"] = score
                est["distance_m"] = self._haversine_distance(
                    latitude, longitude, est["lat"], est["lon"]
                )
                scored_establishments.append(est)
            
            # 5. Ordena por score e aplica diversificação
            final_recommendations = self._diversify_recommendations(
                scored_establishments, max_results
            )
            
            # 6. Enriquece com dados contextuais
            enriched_recommendations = await self._enrich_recommendations(
                final_recommendations, context
            )
            
            # 7. Registra interação para aprendizado
            await self._log_recommendation_interaction(
                user_id, latitude, longitude, query, preferences, enriched_recommendations
            )
            
            return enriched_recommendations
            
        except Exception as e:
            logger.error(f"Erro nas recomendações avançadas: {e}")
            return []
    
    async def _get_establishments_in_area(
        self, lat: float, lon: float, radius_km: float, limit: int
    ) -> List[Dict[str, Any]]:
        """Busca estabelecimentos na área com dados enriquecidos."""
        
        if not self.sb:
            return self._get_mock_establishments(lat, lon, limit)
        
        try:
            # Query otimizada com joins
            lat_delta = radius_km / 111.0
            lon_delta = radius_km / (111.0 * math.cos(math.radians(lat)))
            
            result = await asyncio.to_thread(
                lambda: self.sb.table("establishments")
                .select("""
                    place_id, name, address, types, rating, user_rating_count,
                    price_level, lat, lon, open_now, opening_hours, photo_url,
                    website_url, phone, description, reviews_excerpt,
                    last_recommended_at, recommend_count, avg_model_score,
                    primary_type, business_status, weekday_text,
                    estab_features!inner(has_live_music, is_romantic, is_budget, food_quality),
                    establishment_insights(summary, top_keywords, sentiment, last_30d_metrics)
                """)
                .gte("lat", lat - lat_delta)
                .lte("lat", lat + lat_delta)
                .gte("lon", lon - lon_delta)
                .lte("lon", lon + lon_delta)
                .neq("business_status", "CLOSED_PERMANENTLY")
                .limit(limit)
                .execute()
            )
            
            establishments = result.data or []
            
            # Filtra por distância exata
            filtered = []
            for est in establishments:
                if est.get("lat") and est.get("lon"):
                    distance = self._haversine_distance(lat, lon, est["lat"], est["lon"])
                    if distance <= radius_km * 1000:
                        # Flatten nested data
                        if est.get("estab_features"):
                            est.update(est["estab_features"])
                        if est.get("establishment_insights"):
                            est.update(est["establishment_insights"])
                        filtered.append(est)
            
            return filtered
            
        except Exception as e:
            logger.error(f"Erro ao buscar estabelecimentos: {e}")
            return self._get_mock_establishments(lat, lon, limit)
    
    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Carrega perfil completo do usuário."""
        
        # Cache check
        cache_key = f"user_profile_{user_id}"
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now().timestamp() - timestamp) < self.cache_ttl:
                return cached_data
        
        if not self.sb or user_id.endswith("_mock"):
            profile = self._get_mock_user_profile(user_id)
        else:
            try:
                # Busca perfil básico
                profile_result = await asyncio.to_thread(
                    lambda: self.sb.table("profiles")
                    .select("*")
                    .eq("id", user_id)
                    .limit(1)
                    .execute()
                )
                
                profile = profile_result.data[0] if profile_result.data else {}
                
                # Busca preferências
                prefs_result = await asyncio.to_thread(
                    lambda: self.sb.table("user_preferences")
                    .select("category, value, weight, confidence, source")
                    .eq("user_id", user_id)
                    .eq("is_active", True)
                    .order("weight", desc=True)
                    .limit(50)
                    .execute()
                )
                
                preferences = prefs_result.data or []
                
                # Busca histórico de interações
                interactions_result = await asyncio.to_thread(
                    lambda: self.sb.table("user_place_interactions")
                    .select("place_id, signal, rating_value, liked, created_at")
                    .eq("user_id", user_id)
                    .gte("created_at", (datetime.now() - timedelta(days=90)).isoformat())
                    .order("created_at", desc=True)
                    .limit(200)
                    .execute()
                )
                
                interactions = interactions_result.data or []
                
                # Processa dados
                profile["preferences"] = self._process_user_preferences(preferences)
                profile["interaction_history"] = self._analyze_user_interactions(interactions)
                profile["behavior_patterns"] = self._extract_behavior_patterns(interactions)
                
            except Exception as e:
                logger.error(f"Erro ao carregar perfil do usuário: {e}")
                profile = self._get_mock_user_profile(user_id)
        
        # Cache result
        self.cache[cache_key] = (profile, datetime.now().timestamp())
        return profile
    
    async def _process_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Processa query em linguagem natural com IA."""
        
        if not query or not self.gemini_api_key:
            return {}
        
        try:
            # Prompt otimizado para extração de intenções
            prompt = f"""
            Analise esta consulta de recomendação de lugares e extraia informações estruturadas:
            
            Consulta: "{query}"
            
            Extraia e retorne em JSON:
            {{
                "intent": "tipo de lugar procurado",
                "cuisine": "tipo de culinária se mencionado",
                "atmosphere": "tipo de ambiente desejado",
                "price_range": "faixa de preço se mencionado",
                "features": ["características específicas mencionadas"],
                "time_context": "contexto temporal se mencionado",
                "group_size": "tamanho do grupo se mencionado",
                "keywords": ["palavras-chave importantes"],
                "sentiment": "positivo/neutro/negativo",
                "confidence": 0.0-1.0
            }}
            
            Seja preciso e extraia apenas informações explícitas ou claramente implícitas.
            """
            
            response = await asyncio.to_thread(
                lambda: litellm.completion(
                    model=self.gemini_api_key.startswith("gemini") and "gemini/gemini-1.5-pro" or "gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
            )
            
            content = response.choices[0].message.content
            
            # Extrai JSON da resposta
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
        except Exception as e:
            logger.error(f"Erro no processamento NL: {e}")
        
        return {}
    
    async def _calculate_comprehensive_score(
        self,
        establishment: Dict[str, Any],
        user_profile: Dict[str, Any],
        user_lat: float,
        user_lon: float,
        query: Optional[str] = None,
        processed_query: Optional[Dict] = None,
        preferences: Optional[List[Dict]] = None,
        context: Optional[Dict] = None
    ) -> float:
        """Calcula score abrangente do estabelecimento."""
        
        scores = {}
        
        # 1. Score por rating e popularidade
        scores["rating"] = self._calculate_rating_score(establishment)
        scores["popularity"] = self._calculate_popularity_score(establishment)
        
        # 2. Score por distância
        distance_m = self._haversine_distance(
            user_lat, user_lon, establishment["lat"], establishment["lon"]
        )
        scores["distance"] = self._calculate_distance_score(distance_m)
        
        # 3. Score por preferências do usuário
        scores["user_preferences"] = self._calculate_preference_score(
            establishment, user_profile, preferences
        )
        
        # 4. Score contextual (horário, dia da semana, etc.)
        scores["contextual"] = self._calculate_contextual_score(
            establishment, context
        )
        
        # 5. Score por prova social (interações recentes)
        scores["social_proof"] = self._calculate_social_proof_score(establishment)
        
        # 6. Score por correspondência com query
        if query or processed_query:
            scores["query_match"] = self._calculate_query_match_score(
                establishment, query, processed_query
            )
            # Ajusta pesos quando há query específica
            self.weights["query_match"] = 0.20
            self.weights["user_preferences"] = 0.15
        
        # Calcula score final ponderado
        final_score = sum(
            scores.get(factor, 0) * weight 
            for factor, weight in self.weights.items()
        )
        
        # Aplica boost por features especiais
        final_score += self._calculate_feature_boost(establishment, user_profile)
        
        return min(final_score, 1.0)
    
    def _calculate_rating_score(self, establishment: Dict[str, Any]) -> float:
        """Calcula score baseado no rating."""
        rating = establishment.get("rating", 0)
        review_count = establishment.get("user_rating_count", 0)
        
        if not rating:
            return 0.0
        
        # Score base do rating
        rating_score = rating / 5.0
        
        # Ajuste por número de avaliações (confiabilidade)
        confidence_multiplier = min(review_count / 50.0, 1.0)
        
        return rating_score * (0.5 + 0.5 * confidence_multiplier)
    
    def _calculate_popularity_score(self, establishment: Dict[str, Any]) -> float:
        """Calcula score baseado na popularidade."""
        recommend_count = establishment.get("recommend_count", 0)
        avg_model_score = establishment.get("avg_model_score", 0)
        
        # Score por recomendações anteriores
        recommend_score = min(recommend_count / 100.0, 1.0)
        
        # Score por performance do modelo
        model_score = avg_model_score if avg_model_score else 0.5
        
        return (recommend_score + model_score) / 2.0
    
    def _calculate_distance_score(self, distance_m: float) -> float:
        """Calcula score baseado na distância."""
        # Score decresce exponencialmente com a distância
        if distance_m <= 500:  # Muito perto
            return 1.0
        elif distance_m <= 1000:  # Perto
            return 0.9
        elif distance_m <= 2000:  # Médio
            return 0.7
        elif distance_m <= 5000:  # Longe
            return 0.5
        else:  # Muito longe
            return 0.2
    
    def _calculate_preference_score(
        self, 
        establishment: Dict[str, Any], 
        user_profile: Dict[str, Any],
        explicit_preferences: Optional[List[Dict]] = None
    ) -> float:
        """Calcula score baseado nas preferências do usuário."""
        
        score = 0.0
        matches = 0
        
        # Preferências explícitas da requisição
        if explicit_preferences:
            for pref in explicit_preferences:
                if self._matches_preference(establishment, pref):
                    score += 0.3
                    matches += 1
        
        # Preferências do perfil do usuário
        user_prefs = user_profile.get("preferences", {})
        for category, prefs in user_prefs.items():
            for pref_data in prefs:
                if self._matches_user_preference(establishment, category, pref_data):
                    weight = pref_data.get("weight", 0.5)
                    confidence = pref_data.get("confidence", 0.7)
                    score += weight * confidence * 0.2
                    matches += 1
        
        # Histórico de interações
        interaction_history = user_profile.get("interaction_history", {})
        if interaction_history.get("liked_types"):
            est_types = establishment.get("types", [])
            for liked_type in interaction_history["liked_types"]:
                if liked_type in est_types:
                    score += 0.1
                    matches += 1
        
        # Normaliza por número de matches
        if matches > 0:
            score = score / max(matches, 1)
        
        return min(score, 1.0)
    
    def _calculate_contextual_score(
        self, 
        establishment: Dict[str, Any], 
        context: Optional[Dict] = None
    ) -> float:
        """Calcula score contextual (horário, dia, etc.)."""
        
        score = 0.5  # Score base
        
        if not context:
            context = {}
        
        current_time = datetime.now()
        hour = current_time.hour
        weekday = current_time.weekday()
        
        # Score por horário de funcionamento
        if establishment.get("open_now"):
            score += 0.3
        
        # Score por adequação ao horário
        if 6 <= hour <= 10:  # Manhã
            if any(t in establishment.get("types", []) for t in ["cafe", "bakery", "breakfast"]):
                score += 0.2
        elif 11 <= hour <= 14:  # Almoço
            if any(t in establishment.get("types", []) for t in ["restaurant", "meal_takeaway"]):
                score += 0.2
        elif 18 <= hour <= 23:  # Jantar/Noite
            if any(t in establishment.get("types", []) for t in ["restaurant", "bar", "night_club"]):
                score += 0.2
        
        # Score por dia da semana
        if weekday >= 5:  # Final de semana
            if establishment.get("has_live_music") or establishment.get("is_romantic"):
                score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_social_proof_score(self, establishment: Dict[str, Any]) -> float:
        """Calcula score por prova social."""
        
        metrics = establishment.get("last_30d_metrics", {})
        if not metrics:
            return 0.5
        
        impressions = metrics.get("impressions", 0)
        clicks = metrics.get("clicks", 0)
        favorites = metrics.get("favorites", 0)
        
        # CTR (Click Through Rate)
        ctr = clicks / impressions if impressions > 0 else 0
        
        # Taxa de favoritos
        favorite_rate = favorites / clicks if clicks > 0 else 0
        
        # Score combinado
        score = (ctr * 0.6 + favorite_rate * 0.4)
        
        return min(score, 1.0)
    
    def _calculate_query_match_score(
        self, 
        establishment: Dict[str, Any], 
        query: Optional[str] = None,
        processed_query: Optional[Dict] = None
    ) -> float:
        """Calcula score de correspondência com a query."""
        
        score = 0.0
        
        if query:
            query_lower = query.lower()
            name_lower = establishment.get("name", "").lower()
            types_str = " ".join(establishment.get("types", [])).lower()
            description = establishment.get("description", "").lower()
            
            # Match exato no nome
            if query_lower in name_lower:
                score += 0.4
            
            # Match parcial no nome
            query_words = query_lower.split()
            name_words = name_lower.split()
            common_words = set(query_words) & set(name_words)
            if common_words:
                score += 0.2 * (len(common_words) / len(query_words))
            
            # Match nos tipos
            if any(word in types_str for word in query_words):
                score += 0.2
            
            # Match na descrição
            if description and any(word in description for word in query_words):
                score += 0.1
        
        if processed_query:
            # Match por intenção extraída pela IA
            intent = processed_query.get("intent", "")
            if intent:
                if self._matches_intent(establishment, intent):
                    score += 0.3
            
            # Match por características
            features = processed_query.get("features", [])
            for feature in features:
                if self._matches_feature(establishment, feature):
                    score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_feature_boost(
        self, 
        establishment: Dict[str, Any], 
        user_profile: Dict[str, Any]
    ) -> float:
        """Calcula boost por features especiais."""
        
        boost = 0.0
        
        # Boost por qualidade da comida
        food_quality = establishment.get("food_quality", 0)
        if food_quality and food_quality > 4.0:
            boost += 0.05
        
        # Boost por features especiais
        if establishment.get("has_live_music"):
            boost += 0.03
        
        if establishment.get("is_romantic"):
            boost += 0.03
        
        # Boost por adequação ao orçamento
        price_level = establishment.get("price_level", "")
        user_budget = user_profile.get("budget_level", 2)
        
        if price_level == "PRICE_LEVEL_INEXPENSIVE" and user_budget <= 1:
            boost += 0.05
        elif price_level == "PRICE_LEVEL_EXPENSIVE" and user_budget >= 3:
            boost += 0.05
        
        return boost
    
    def _diversify_recommendations(
        self, 
        establishments: List[Dict[str, Any]], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Aplica diversificação nas recomendações."""
        
        # Ordena por score
        establishments.sort(key=lambda x: x["score"], reverse=True)
        
        # Aplica diversificação por tipo
        diversified = []
        type_counts = defaultdict(int)
        max_per_type = max(2, max_results // 4)
        
        for est in establishments:
            if len(diversified) >= max_results:
                break
            
            primary_type = est.get("primary_type", "restaurant")
            
            if type_counts[primary_type] < max_per_type:
                diversified.append(est)
                type_counts[primary_type] += 1
        
        # Preenche com os melhores restantes se necessário
        remaining_slots = max_results - len(diversified)
        if remaining_slots > 0:
            for est in establishments:
                if est not in diversified and remaining_slots > 0:
                    diversified.append(est)
                    remaining_slots -= 1
        
        return diversified
    
    async def _enrich_recommendations(
        self, 
        recommendations: List[Dict[str, Any]], 
        context: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Enriquece recomendações com dados adicionais."""
        
        for rec in recommendations:
            # Adiciona informações de horário de funcionamento
            if rec.get("opening_hours"):
                rec["is_open_now"] = self._is_currently_open(rec["opening_hours"])
            
            # Adiciona estimativa de tempo de espera
            rec["estimated_wait_time"] = self._estimate_wait_time(rec)
            
            # Adiciona razão da recomendação
            rec["recommendation_reason"] = self._generate_recommendation_reason(rec)
            
            # Adiciona tags personalizadas
            rec["tags"] = self._generate_tags(rec)
        
        return recommendations
    
    async def _log_recommendation_interaction(
        self,
        user_id: str,
        latitude: float,
        longitude: float,
        query: Optional[str],
        preferences: Optional[List[Dict]],
        recommendations: List[Dict[str, Any]]
    ):
        """Registra interação para aprendizado."""
        
        if not self.sb or user_id.endswith("_mock"):
            return
        
        try:
            interaction_data = {
                "user_id": user_id.replace("user_", ""),
                "signal": "recommendation",
                "query_text": query,
                "user_lat": latitude,
                "user_lon": longitude,
                "features": {
                    "algorithm_version": "3.0",
                    "preferences": preferences,
                    "result_count": len(recommendations),
                    "avg_score": sum(r["score"] for r in recommendations) / len(recommendations) if recommendations else 0,
                    "top_score": max(r["score"] for r in recommendations) if recommendations else 0
                }
            }
            
            await asyncio.to_thread(
                lambda: self.sb.table("user_place_interactions")
                .insert(interaction_data)
                .execute()
            )
            
            # Atualiza contadores dos estabelecimentos
            for rec in recommendations:
                await asyncio.to_thread(
                    lambda: self.sb.table("establishments")
                    .update({
                        "recommend_count": rec.get("recommend_count", 0) + 1,
                        "last_recommended_at": datetime.now(timezone.utc).isoformat()
                    })
                    .eq("place_id", rec["place_id"])
                    .execute()
                )
                
        except Exception as e:
            logger.error(f"Erro ao logar interação: {e}")
    
    # --- Métodos auxiliares ---
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calcula distância haversine em metros."""
        R = 6371000  # Raio da Terra em metros
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        return R * c
    
    def _matches_preference(self, establishment: Dict[str, Any], preference: Dict) -> bool:
        """Verifica se estabelecimento corresponde à preferência."""
        pref_value = preference.get("value", "").lower()
        
        name = establishment.get("name", "").lower()
        types = " ".join(establishment.get("types", [])).lower()
        description = establishment.get("description", "").lower()
        
        return (pref_value in name or 
                pref_value in types or 
                pref_value in description)
    
    def _matches_user_preference(
        self, 
        establishment: Dict[str, Any], 
        category: str, 
        pref_data: Dict
    ) -> bool:
        """Verifica match com preferência do usuário."""
        pref_value = pref_data.get("value", "").lower()
        
        if category == "food":
            types = establishment.get("types", [])
            return any("restaurant" in t or "food" in t for t in types) and pref_value in str(types).lower()
        
        elif category == "music":
            return establishment.get("has_live_music", False) and "música" in pref_value
        
        elif category == "culture":
            types = establishment.get("types", [])
            return any(t in ["museum", "art_gallery", "library"] for t in types)
        
        return False
    
    def _matches_intent(self, establishment: Dict[str, Any], intent: str) -> bool:
        """Verifica se estabelecimento corresponde à intenção."""
        intent_lower = intent.lower()
        types = " ".join(establishment.get("types", [])).lower()
        name = establishment.get("name", "").lower()
        
        return intent_lower in types or intent_lower in name
    
    def _matches_feature(self, establishment: Dict[str, Any], feature: str) -> bool:
        """Verifica se estabelecimento tem a feature."""
        feature_lower = feature.lower()
        
        if "música" in feature_lower or "music" in feature_lower:
            return establishment.get("has_live_music", False)
        
        if "romântico" in feature_lower or "romantic" in feature_lower:
            return establishment.get("is_romantic", False)
        
        if "barato" in feature_lower or "budget" in feature_lower:
            return establishment.get("is_budget", False)
        
        return False
    
    def _process_user_preferences(self, preferences: List[Dict]) -> Dict[str, List[Dict]]:
        """Processa preferências do usuário por categoria."""
        processed = defaultdict(list)
        
        for pref in preferences:
            category = pref.get("category", "other")
            processed[category].append(pref)
        
        return dict(processed)
    
    def _analyze_user_interactions(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Analisa histórico de interações do usuário."""
        analysis = {
            "total_interactions": len(interactions),
            "liked_places": [],
            "disliked_places": [],
            "liked_types": [],
            "avg_rating": 0,
            "interaction_patterns": {}
        }
        
        ratings = []
        type_counter = Counter()
        
        for interaction in interactions:
            signal = interaction.get("signal", "")
            place_id = interaction.get("place_id", "")
            
            if signal == "favorite" or interaction.get("liked"):
                analysis["liked_places"].append(place_id)
            
            if signal == "dislike" or interaction.get("liked") == False:
                analysis["disliked_places"].append(place_id)
            
            if interaction.get("rating_value"):
                ratings.append(interaction["rating_value"])
            
            # Conta tipos de lugares interagidos (seria necessário join com establishments)
            # Por simplicidade, vamos pular esta parte
        
        if ratings:
            analysis["avg_rating"] = sum(ratings) / len(ratings)
        
        return analysis
    
    def _extract_behavior_patterns(self, interactions: List[Dict]) -> Dict[str, Any]:
        """Extrai padrões de comportamento do usuário."""
        patterns = {
            "preferred_times": [],
            "preferred_days": [],
            "interaction_frequency": 0,
            "exploration_tendency": 0  # 0-1, onde 1 = muito explorador
        }
        
        # Análise temporal
        hours = []
        days = []
        
        for interaction in interactions:
            created_at = interaction.get("created_at", "")
            if created_at:
                try:
                    dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                    hours.append(dt.hour)
                    days.append(dt.weekday())
                except:
                    continue
        
        if hours:
            patterns["preferred_times"] = list(set(hours))
        
        if days:
            patterns["preferred_days"] = list(set(days))
        
        # Frequência de interação
        if interactions:
            time_span = (datetime.now() - datetime.fromisoformat(
                interactions[-1]["created_at"].replace('Z', '+00:00')
            )).days
            patterns["interaction_frequency"] = len(interactions) / max(time_span, 1)
        
        return patterns
    
    def _is_currently_open(self, opening_hours: Dict) -> bool:
        """Verifica se estabelecimento está aberto agora."""
        # Implementação simplificada
        return True  # Por enquanto, assume que está aberto
    
    def _estimate_wait_time(self, establishment: Dict[str, Any]) -> str:
        """Estima tempo de espera."""
        rating = establishment.get("rating", 0)
        review_count = establishment.get("user_rating_count", 0)
        
        if rating > 4.5 and review_count > 100:
            return "15-30 min"
        elif rating > 4.0:
            return "5-15 min"
        else:
            return "< 5 min"
    
    def _generate_recommendation_reason(self, establishment: Dict[str, Any]) -> str:
        """Gera razão da recomendação."""
        reasons = []
        
        if establishment.get("rating", 0) > 4.5:
            reasons.append("Muito bem avaliado")
        
        if establishment.get("has_live_music"):
            reasons.append("Música ao vivo")
        
        if establishment.get("is_romantic"):
            reasons.append("Ambiente romântico")
        
        if establishment.get("distance_m", 0) < 500:
            reasons.append("Muito próximo")
        
        return ", ".join(reasons) if reasons else "Recomendado para você"
    
    def _generate_tags(self, establishment: Dict[str, Any]) -> List[str]:
        """Gera tags personalizadas."""
        tags = []
        
        if establishment.get("rating", 0) > 4.5:
            tags.append("⭐ Top rated")
        
        if establishment.get("has_live_music"):
            tags.append("🎵 Live music")
        
        if establishment.get("is_romantic"):
            tags.append("💕 Romantic")
        
        if establishment.get("is_budget"):
            tags.append("💰 Budget friendly")
        
        if establishment.get("open_now"):
            tags.append("🟢 Open now")
        
        return tags
    
    def _get_mock_establishments(self, lat: float, lon: float, limit: int) -> List[Dict[str, Any]]:
        """Retorna estabelecimentos mock para desenvolvimento."""
        return [
            {
                "place_id": f"mock_place_{i}",
                "name": f"Restaurante Mock {i}",
                "address": f"Rua Mock {i}, São Paulo",
                "types": ["restaurant", "food"],
                "rating": 4.0 + (i % 10) / 10,
                "user_rating_count": 50 + i * 10,
                "price_level": "PRICE_LEVEL_MODERATE",
                "lat": lat + (i - limit/2) * 0.001,
                "lon": lon + (i - limit/2) * 0.001,
                "open_now": i % 2 == 0,
                "has_live_music": i % 3 == 0,
                "is_romantic": i % 4 == 0,
                "food_quality": 4.0 + (i % 5) / 5,
                "recommend_count": i * 5,
                "avg_model_score": 0.7 + (i % 3) / 10
            }
            for i in range(limit)
        ]
    
    def _get_mock_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Retorna perfil mock do usuário."""
        return {
            "id": user_id,
            "preferences": {
                "food": [{"value": "sushi", "weight": 0.8, "confidence": 0.9}],
                "music": [{"value": "jazz", "weight": 0.7, "confidence": 0.8}]
            },
            "interaction_history": {
                "total_interactions": 50,
                "liked_places": ["place1", "place2"],
                "liked_types": ["restaurant", "bar"],
                "avg_rating": 4.2
            },
            "behavior_patterns": {
                "preferred_times": [19, 20, 21],
                "preferred_days": [5, 6],
                "interaction_frequency": 0.5
            },
            "budget_level": 2
        }

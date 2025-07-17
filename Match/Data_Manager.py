# ==========================================
# GESTIÓN DE CACHE PERSISTENTE
# ==========================================

import multiprocessing as mp
import pickle
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from functools import partial

class GeocodingCache:
    """Clase para manejar el cache persistente de geocoding"""
    
    def __init__(self, cache_file="geocoding_cache.pkl"):
        self.cache_file = cache_file
        self.cache = self._cargar_cache()
        self.cambios_sin_guardar = 0
    
    def _cargar_cache(self):
        """Carga el cache desde archivo"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    cache = pickle.load(f)
                print(f"Cache cargado: {len(cache)} entradas existentes")
                return cache
            except Exception as e:
                print(f"Error cargando cache: {e}")
                return {}
        return {}
    
    def guardar_cache(self):
        """Guarda el cache a archivo"""
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            print(f"Cache guardado: {len(self.cache)} entradas")
            self.cambios_sin_guardar = 0
        except Exception as e:
            print(f"Error guardando cache: {e}")
    
    def get(self, key):
        """Obtiene valor del cache"""
        return self.cache.get(key)
    
    def set(self, key, value):
        """Establece valor en cache"""
        self.cache[key] = value
        self.cambios_sin_guardar += 1
        
        # Auto-guardar cada 50 cambios
        if self.cambios_sin_guardar >= 50:
            self.guardar_cache()
    
    def __len__(self):
        return len(self.cache)

# ==========================================
# FUNCIONES BÁSICAS (SIN GEOCODING)
# ==========================================
def obtener_etiqueta_por_cp(cp_value):
    """Obtiene etiqueta basada en CP - Solo acepta 4 o 5 cifras"""
    if pd.isna(cp_value):
        return None
    
    try:
        # Conversión más directa
        cp_str = str(cp_value).replace('.0', '').strip()
        if not cp_str or cp_str in ('', 'nan'):
            return None
        
        # Validar que tenga exactamente 4 o 5 cifras
        if len(cp_str) not in [4, 5]:
            return None
        
        # Validar que sean solo dígitos
        if not cp_str.isdigit():
            return None
        
        cp_numeric = int(cp_str)
        
        # Rangos optimizados
        if 1000 <= cp_numeric < 17000:
            return 'cdmx'
        elif 50000 <= cp_numeric <= 57999:
            return 'edomex'
        else:
            return 'foraneo'
    except (ValueError, TypeError):
        return None
def obtener_etiqueta_por_cp_index(cp_index_value):
    """Obtiene etiqueta basada en CP_INDEX - Reutiliza función CP"""
    return obtener_etiqueta_por_cp(cp_index_value)

def validar_con_codigos_postales(row):
    """
    Valida y asigna entidad SOLO usando códigos postales CP y CP_INDEX
    
    Returns:
        dict: {
            'entidad': str,
            'metodo': str,
            'necesita_geocoding': bool,
            'detalle': str
        }
    """
    resultado = {
        'entidad': row.get('ENTIDAD', ''),
        'metodo': 'sin_cambio',
        'necesita_geocoding': False,
        'detalle': ''
    }
    
    # Obtener etiquetas de códigos postales
    etiqueta_cp = obtener_etiqueta_por_cp(row['CP'])
    etiqueta_cp_index = obtener_etiqueta_por_cp_index(row['CP_INDEX'])
    
    # CASO 1: CP_INDEX es nulo o vacío
    if etiqueta_cp_index is None:
        if etiqueta_cp is not None:
            resultado.update({
                'entidad': etiqueta_cp,
                'metodo': 'cp_solo',
                'detalle': f'Solo CP válido: {etiqueta_cp}'
            })
        else:
            resultado.update({
                'metodo': 'cp_invalido',
                'necesita_geocoding': True,
                'detalle': 'CP y CP_INDEX inválidos - requiere geocoding'
            })
    
    # CASO 2: Ambos CP y CP_INDEX tienen valores
    else:
        # CASO 2a: CP y CP_INDEX son iguales
        if etiqueta_cp == etiqueta_cp_index:
            resultado.update({
                'entidad': etiqueta_cp,
                'metodo': 'cp_iguales',
                'detalle': f'CP = CP_INDEX = {etiqueta_cp}'
            })
        
        # CASO 2b: CP y CP_INDEX son diferentes
        else:
            resultado.update({
                'metodo': 'cp_diferentes',
                'necesita_geocoding': True,
                'detalle': f'CP:{etiqueta_cp} ≠ CP_INDEX:{etiqueta_cp_index} - requiere geocoding'
            })
    
    return resultado
# ==========================================
# FUNCIONES DE GEOCODING MULTIPROCESO
# ==========================================

def geocoding_worker(coordenadas_batch):
    """
    Worker function para geocoding en paralelo
    
    Args:
        coordenadas_batch: Lista de tuplas (index, lat, lng)
    
    Returns:
        dict: {index: etiqueta}
    """
    from geopy.geocoders import Nominatim
    from geopy.extra.rate_limiter import RateLimiter
    import time
    
    # Inicializar geocodificador para este proceso
    geolocator = Nominatim(
        user_agent=f"entidad_detector_worker_{os.getpid()}", 
        timeout=15
    )
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=0.1)
    
    resultados = {}
    
    for idx, lat, lng in coordenadas_batch:
        if pd.isna(lat) or pd.isna(lng):
            resultados[idx] = None
            continue
        
        # Crear clave para cache
        key = (round(float(lat), 4), round(float(lng), 4))
        
        try:
            location = reverse(f"{lat}, {lng}", language='es')
            
            if location and location.raw:
                address = location.raw.get('address', {})
                state = address.get('state', '').lower()
                
                if any(term in state for term in ['ciudad de méxico', 'cdmx', 'distrito federal']):
                    etiqueta = 'cdmx'
                elif any(term in state for term in ['méxico', 'estado de méxico']):
                    etiqueta = 'edomex'
                else:
                    etiqueta = 'foraneo'
            else:
                etiqueta = None
            
            resultados[idx] = etiqueta
            
        except Exception as e:
            resultados[idx] = None
    
    return resultados

def procesar_geocoding_paralelo(coordenadas_pendientes, max_workers=True):
    """
    Procesa geocoding en paralelo con múltiples procesos
    
    Args:
        coordenadas_pendientes: Lista de tuplas (index, lat, lng)
        max_workers: Número máximo de procesos
    
    Returns:
        dict: {index: etiqueta}
    """
    if not coordenadas_pendientes:
        return {}
    
    if max_workers is None:
        max_workers = min(4, mp.cpu_count())  # Máximo 4 procesos para no sobrecargar APIs
    
    # Dividir trabajo en lotes
    batch_size = max(1, len(coordenadas_pendientes) // max_workers)
    batches = [
        coordenadas_pendientes[i:i + batch_size] 
        for i in range(0, len(coordenadas_pendientes), batch_size)
    ]
    
    print(f"Procesando {len(coordenadas_pendientes)} coordenadas en {len(batches)} lotes con {max_workers} procesos")
    
    resultados_finales = {}
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Enviar trabajos
        future_to_batch = {executor.submit(geocoding_worker, batch): batch for batch in batches}
        
        # Recoger resultados
        for i, future in enumerate(as_completed(future_to_batch)):
            try:
                resultados_batch = future.result()
                resultados_finales.update(resultados_batch)
                print(f"Completado lote {i+1}/{len(batches)} - {len(resultados_batch)} coordenadas procesadas")
            except Exception as e:
                print(f"Error en lote {i+1}: {e}")
    
    return resultados_finales
def aplicar_geocoding_con_cache(df_pendientes):
    """
    Aplica geocoding usando cache y multiproceso para coordenadas no cacheadas
    
    Args:
        df_pendientes: DataFrame con registros que necesitan geocoding
    
    Returns:
        dict: {index: etiqueta}
    """
    global geo_cache
    
    resultados = {}
    coordenadas_pendientes = []
    
    print(f"Verificando cache para {len(df_pendientes)} registros...")
    
    # Verificar cuáles están en cache
    for idx, row in df_pendientes.iterrows():
        lat, lng = row.get('lat_centro'), row.get('lng_centro')
        
        if pd.isna(lat) or pd.isna(lng):
            resultados[idx] = None
            continue
        
        # Crear clave para cache
        key = (round(float(lat), 4), round(float(lng), 4))
        
        # Verificar cache
        cached_result = geo_cache.get(key)
        if cached_result is not None:
            resultados[idx] = cached_result
        else:
            coordenadas_pendientes.append((idx, lat, lng))
    
    print(f"Cache hits: {len(resultados)}, Geocoding pendiente: {len(coordenadas_pendientes)}")
    
    # Procesar coordenadas pendientes en paralelo
    if coordenadas_pendientes:
        resultados_geocoding = procesar_geocoding_paralelo(coordenadas_pendientes)
        
        # Agregar resultados al cache
        for idx, etiqueta in resultados_geocoding.items():
            # Obtener coordenadas originales
            for coord_idx, lat, lng in coordenadas_pendientes:
                if coord_idx == idx:
                    key = (round(float(lat), 4), round(float(lng), 4))
                    geo_cache.set(key, etiqueta)
                    break
        
        resultados.update(resultados_geocoding)
    
    # Guardar cache
    geo_cache.guardar_cache()
    
    return resultados
# ==========================================
# FUNCIÓN PRINCIPAL OPTIMIZADA
# ==========================================

def procesar_entidades_optimizado(df, usar_geocoding, max_workers=True):
    """
    Procesa entidades de forma optimizada con multiproceso opcional
    
    Args:
        df: DataFrame a procesar
        usar_geocoding: None (desactivado), True (activado), False (solo CP)
        max_workers: Número de procesos para geocoding
    
    Returns:
        tuple: (df_procesado, estadisticas)
    """
    df_copy = df.copy()
    
    print(f"Procesando {len(df_copy)} registros...")
    print(f"Geocoding: {'Desactivado' if usar_geocoding is None else 'Activado' if usar_geocoding else 'Solo CP'}")
    
    # ETAPA 1: Validación con códigos postales (rápido)
    print("\n=== ETAPA 1: Validación con códigos postales ===")
    
    validaciones = []
    for idx, row in df_copy.iterrows():
        validacion = validar_con_codigos_postales(row)
        validacion['index'] = idx
        validaciones.append(validacion)
    
    # Aplicar resultados de validación CP
    registros_resueltos = 0
    registros_pendientes = []
    
    for validacion in validaciones:
        idx = validacion['index']
        
        if not validacion['necesita_geocoding']:
            # Actualizar directamente
            df_copy.loc[idx, 'ENTIDAD'] = validacion['entidad']
            registros_resueltos += 1
        else:
            # Marcar para geocoding
            if usar_geocoding is True:
                registros_pendientes.append(idx)
            # Si usar_geocoding es None o False, mantener valor original
    
    print(f"Resueltos con CP: {registros_resueltos}/{len(df_copy)} ({(registros_resueltos/len(df_copy)*100):.1f}%)")
    print(f"Pendientes geocoding: {len(registros_pendientes)}")
    
    # ETAPA 2: Geocoding (si está activado)
    if usar_geocoding is True and registros_pendientes:
        print(f"\n=== ETAPA 2: Geocoding de {len(registros_pendientes)} registros ===")
        
        df_pendientes = df_copy.loc[registros_pendientes]
        resultados_geocoding = aplicar_geocoding_con_cache(df_pendientes)
        
        # Aplicar resultados de geocoding
        for idx, etiqueta_geo in resultados_geocoding.items():
            if etiqueta_geo is not None:
                # Re-validar con geocoding
                row = df_copy.loc[idx]
                etiqueta_cp = obtener_etiqueta_por_cp(row['CP'])
                etiqueta_cp_index = obtener_etiqueta_por_cp_index(row['CP_INDEX'])
                
                # Lógica de decisión con geocoding
                if etiqueta_cp_index is None:
                    if etiqueta_cp == etiqueta_geo:
                        df_copy.loc[idx, 'ENTIDAD'] = etiqueta_cp
                else:
                    if etiqueta_cp == etiqueta_geo:
                        df_copy.loc[idx, 'ENTIDAD'] = etiqueta_cp
                    elif etiqueta_cp_index == etiqueta_geo:
                        df_copy.loc[idx, 'ENTIDAD'] = etiqueta_cp_index
    
    # Estadísticas finales
    estadisticas = {
        'total_procesados': len(df_copy),
        'resueltos_con_cp': registros_resueltos,
        'procesados_geocoding': len(registros_pendientes) if usar_geocoding is True else 0,
        'distribucion_final': df_copy['ENTIDAD'].value_counts().to_dict()
    }
    
    return df_copy, estadisticas

def aplicar_procesamiento_completo(datasets_dict, usar_geocoding, max_workers=True):
    """
    Aplica el procesamiento completo a todos los datasets
    
    Args:
        datasets_dict: Diccionario con datasets
        usar_geocoding: None (desactivado), True (activado), False (solo CP)
        max_workers: Número de procesos para geocoding
    """
    datasets_actualizados = {}
    estadisticas_globales = {}
    
    tiempo_inicio = time.time()
    
    for year in ["2020", "2021", "2022", "2023"]:
        dataset_name = f"df_coordenadas_sustentantes_{year}"
        
        print(f"\n{'='*70}")
        print(f"Procesando dataset: {dataset_name}")
        print(f"{'='*70}")
        
        df = datasets_dict[dataset_name].copy()
        
        # Procesar dataset
        df_procesado, estadisticas = procesar_entidades_optimizado(
            df, 
            usar_geocoding=usar_geocoding,
            max_workers=max_workers
        )
        
        # Guardar resultados
        datasets_actualizados[dataset_name] = df_procesado
        estadisticas_globales[year] = estadisticas
        
        # Mostrar resumen
        print(f"\n--- Resumen {year} ---")
        print(f"Total: {estadisticas['total_procesados']}")
        print(f"Resueltos con CP: {estadisticas['resueltos_con_cp']}")
        print(f"Geocoding: {estadisticas['procesados_geocoding']}")
        print(f"Distribución final: {estadisticas['distribucion_final']}")
    
    tiempo_total = time.time() - tiempo_inicio
    print(f"\n{'='*70}")
    print(f"PROCESAMIENTO COMPLETADO EN {tiempo_total:.2f} segundos")
    print(f"{'='*70}")
    
    return datasets_actualizados, estadisticas_globales

# ==========================================
# EJECUTAR PROCESAMIENTO
# ==========================================

print("=== PROCESAMIENTO DE ENTIDADES OPTIMIZADO ===")
print("Opciones de geocoding:")
print("- usar_geocoding=None  : Solo códigos postales (MÁS RÁPIDO)")
print("- usar_geocoding=False : Solo códigos postales") 
print("- usar_geocoding=True  : CP + Geocoding multiproceso")

# OPCIÓN 1: Solo códigos postales (recomendado para empezar)
datasets_procesados, estadisticas = aplicar_procesamiento_completo(
    original_datasets, 
    usar_geocoding=None,  # Cambiar a True para activar geocoding
    max_workers=10
)

# Actualizar datasets originales
for key, df in datasets_procesados.items():
    original_datasets[key] = df

print("\n✅ Procesamiento completado!")
```Python
import pandas as pd
import numpy as np
import geopandas as gpd
import unicodedata
import fiona
import os
from typing import Dict, Optional, Union, List, Tuple

class LoadData:
    """
    Clase para cargar y procesar los datasets del proyecto COMIPEMS.
    
    Proporciona funciones para:
    - Cargar datasets originales, clasificados y procesados
    - Normalizar textos (minúsculas, sin acentos)
    - Estandarizar formatos de datos (códigos postales, nombres de entidades)
    - Cargar datos geoespaciales
    """
    
    def __init__(self, base_path: str = "."):
        """
        Inicializa la clase con las rutas de los directorios de datos.
        
        Args:
            base_path: Ruta base del proyecto
        """
        self.base_path = base_path
        self.data_dir = os.path.join(base_path, "DataSets_Iniciales/coordenadas_obtenidas_google_maps/")
        self.data_dir_classified = os.path.join(self.data_dir, "Coordernadas_obtenidas_clasificadas/")
        self.data_dir_empty = os.path.join(self.data_dir_classified, "nulos/")
        self.data_dir_different = os.path.join(self.data_dir_classified, "diferentes/")
        self.data_dir_correct = os.path.join(self.data_dir_classified, "iguales/")
        self.data_dir_cncp = os.path.join(base_path, "CP_CatalogoNacionaldeCodigosPostales/")
        self.data_dir_gpkg = os.path.join(base_path, "DataSets_Iniciales/colonias_indice_marginacion_cdmx/")
        
        # Mapeo de nombres de entidades
        self.entity_mapping = {
            "mexico": "edomex",
            "ciudad de mexico": "cdmx"
        }
        
        # Años disponibles en el dataset
        self.available_years = ["2020", "2021", "2022", "2023"]
    
    def minusculas_sin_acentos(self, texto: Union[str, float, int]) -> str:
        """
        Convierte texto a minúsculas y elimina acentos.
        
        Args:
            texto: Texto a normalizar
            
        Returns:
            Texto normalizado en minúsculas y sin acentos
        """
        if pd.isna(texto):
            return ""
        
        texto = str(texto).lower()
        # Separar caracteres base de las tildes
        texto = unicodedata.normalize('NFD', texto)
        # Descartar todos los caracteres de marca
        texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
        return texto
    
    def _normalize_text_columns(self, df: pd.DataFrame, 
                               columns: List[str]) -> pd.DataFrame:
        """
        Normaliza las columnas de texto especificadas (minúsculas y sin acentos).
        
        Args:
            df: DataFrame a procesar
            columns: Lista de columnas a normalizar
            
        Returns:
            DataFrame con las columnas normalizadas
        """
        df_copy = df.copy()
        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].apply(self.minusculas_sin_acentos)
        return df_copy
    
    def _convert_postal_code_to_int(self, df: pd.DataFrame, 
                                   postal_code_columns: List[str]) -> pd.DataFrame:
        """
        Convierte las columnas de código postal a tipo int64.
        
        Args:
            df: DataFrame a procesar
            postal_code_columns: Lista de columnas de código postal
            
        Returns:
            DataFrame con las columnas de código postal convertidas a int64
        """
        df_copy = df.copy()
        for col in postal_code_columns:
            if col in df_copy.columns:
                # Convertir solo si no es NaN
                mask = df_copy[col].notna()
                if mask.any():
                    df_copy.loc[mask, col] = df_copy.loc[mask, col].astype(int)
        return df_copy
    
    def _remap_entity_names(self, df: pd.DataFrame, 
                           entity_column: str) -> pd.DataFrame:
        """
        Remapea los nombres de entidades según el diccionario de mapeo.
        
        Args:
            df: DataFrame a procesar
            entity_column: Nombre de la columna de entidades
            
        Returns:
            DataFrame con los nombres de entidades remapeados
        """
        df_copy = df.copy()
        if entity_column in df_copy.columns:
            df_copy[entity_column] = df_copy[entity_column].apply(
                lambda x: self.entity_mapping.get(self.minusculas_sin_acentos(x), x) 
                if pd.notna(x) else x
            )
        return df_copy
    
    def load_original_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets originales sin ninguna modificación.
        
        Returns:
            Diccionario de DataFrames originales por año
        """
        datasets = {}
        
        for year in self.available_years:
            file_path = os.path.join(self.data_dir, f"coordenadas_sustentantes_{year}.csv")
            try:
                datasets[f"df_coordenadas_sustentantes_{year}"] = pd.read_csv(file_path)
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo {file_path}")
                
        return datasets
    
    def load_classified_null_datasets(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets donde el código postal API es nulo.
        
        Args:
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            Diccionario de DataFrames con códigos postales nulos por año
        """
        datasets = {}
        
        for year in self.available_years:
            file_path = os.path.join(self.data_dir_empty, f"df_coordenadas_sustentantes_{year}_nulos.csv")
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                if normalize:
                    # Normalizar columnas de texto
                    df = self._normalize_text_columns(df, ["SUS_COL", "SUS_DEL", "NOM_ENT"])
                    # Remapear nombres de entidades
                    df = self._remap_entity_names(df, "NOM_ENT")
                    # Convertir códigos postales a int64 donde sea posible
                    df = self._convert_postal_code_to_int(df, ["SUS_CP", "postal_code_catalogo"])
                
                datasets[f"df_coordenadas_sustentantes_{year}_nulos"] = df
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo {file_path}")
                
        return datasets
    
    def load_classified_equal_datasets(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets donde el código postal API es igual al SUS_CP.
        
        Args:
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            Diccionario de DataFrames con códigos postales iguales por año
        """
        datasets = {}
        
        for year in self.available_years:
            file_path = os.path.join(self.data_dir_correct, f"df_coordenadas_sustentantes_{year}_iguales.csv")
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                if normalize:
                    # Normalizar columnas de texto
                    df = self._normalize_text_columns(df, ["SUS_COL", "SUS_DEL", "NOM_ENT"])
                    # Remapear nombres de entidades
                    df = self._remap_entity_names(df, "NOM_ENT")
                    # Convertir códigos postales a int64
                    df = self._convert_postal_code_to_int(df, ["SUS_CP", "postal_code_api"])
                
                datasets[f"df_coordenadas_sustentantes_{year}_iguales"] = df
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo {file_path}")
                
        return datasets
    
    def load_classified_different_datasets(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets donde el código postal API es diferente del SUS_CP.
        
        Args:
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            Diccionario de DataFrames con códigos postales diferentes por año
        """
        datasets = {}
        
        for year in self.available_years:
            file_path = os.path.join(self.data_dir_different, f"df_coordenadas_sustentantes_{year}_diferentes.csv")
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                if normalize:
                    # Normalizar columnas de texto
                    df = self._normalize_text_columns(df, ["SUS_COL", "SUS_DEL", "NOM_ENT"])
                    # Remapear nombres de entidades
                    df = self._remap_entity_names(df, "NOM_ENT")
                    # Convertir códigos postales a int64
                    df = self._convert_postal_code_to_int(df, ["SUS_CP", "postal_code_api"])
                
                datasets[f"df_coordenadas_sustentantes_{year}_diferentes"] = df
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo {file_path}")
                
        return datasets
    
    def load_processed_datasets(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets procesados con códigos postales completados.
        
        Args:
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            Diccionario de DataFrames procesados por año
        """
        datasets = {}
        
        for year in self.available_years:
            file_path = os.path.join(self.data_dir_empty, f"df_coordenadas_sustentantes_{year}_completos.csv")
            try:
                df = pd.read_csv(file_path, encoding="utf-8")
                
                if normalize:
                    # Normalizar columnas de texto
                    df = self._normalize_text_columns(df, ["SUS_COL", "SUS_DEL", "NOM_ENT"])
                    # Remapear nombres de entidades
                    df = self._remap_entity_names(df, "NOM_ENT")
                    # Convertir códigos postales a int64 donde sea posible
                    df = self._convert_postal_code_to_int(df, ["SUS_CP", "postal_code_catalogo"])
                
                datasets[f"df_coordenadas_sustentantes_{year}_completos"] = df
            except FileNotFoundError:
                print(f"Advertencia: No se encontró el archivo {file_path}")
                
        return datasets
    
    def load_still_empty_datasets(self, normalize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Carga los datasets procesados pero que aún tienen valores nulos en código postal.
        
        Args:
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            Diccionario de DataFrames con valores aún nulos por año
        """
        datasets = {}
        processed_datasets = self.load_processed_datasets(normalize=False)
        
        for name, df in processed_datasets.items():
            # Filtrar solo filas donde el código postal sigue siendo nulo
            df_still_null = df[df['postal_code_catalogo'].isna()].copy()
            
            if normalize:
                # Normalizar columnas de texto
                df_still_null = self._normalize_text_columns(df_still_null, ["SUS_COL", "SUS_DEL", "NOM_ENT"])
                # Remapear nombres de entidades
                df_still_null = self._remap_entity_names(df_still_null, "NOM_ENT")
                # Convertir códigos postales a int64 donde sea posible
                df_still_null = self._convert_postal_code_to_int(df_still_null, ["SUS_CP"])
            
            # Cambiar el nombre para indicar que son los que siguen vacíos
            year = name.split('_')[-2]
            datasets[f"df_coordenadas_sustentantes_{year}_still_null"] = df_still_null
                
        return datasets
    
    def load_postal_codes_catalog(self, normalize: bool = True) -> pd.DataFrame:
        """
        Carga el catálogo unificado de códigos postales.
        
        Args:
            normalize: Si es True, normaliza textos
            
        Returns:
            DataFrame con el catálogo unificado de códigos postales
        """
        file_path = os.path.join(self.data_dir_cncp, "df_cncp_unificado.csv")
        try:
            df = pd.read_csv(file_path, encoding="utf-8")
            
            if normalize:
                # Normalizar columnas de texto
                df = self._normalize_text_columns(df, ["d_asenta", "D_mnpio", "d_estado"])
                # Remapear nombres de entidades
                df = self._remap_entity_names(df, "d_estado")
                # Convertir código postal a int64
                df = self._convert_postal_code_to_int(df, ["d_codigo", "c_CP"])
            
            return df
        except FileNotFoundError:
            print(f"Advertencia: No se encontró el archivo {file_path}")
            return pd.DataFrame()
    
    def load_marginacion_gdf(self, as_dataframe: bool = False, normalize: bool = True) -> Union[gpd.GeoDataFrame, pd.DataFrame]:
        """
        Carga el GeoDataFrame con el índice de marginación.
        
        Args:
            as_dataframe: Si es True, devuelve un pandas DataFrame (sin geometría)
            normalize: Si es True, normaliza textos y convierte códigos postales
            
        Returns:
            GeoDataFrame o DataFrame con el índice de marginación
        """
        file_path = os.path.join(self.data_dir_gpkg, "indice_marginacion_colonias_valle_mexico.gpkg")
        try:
            # Listar las capas disponibles en el archivo GPKG
            name_layer = fiona.listlayers(file_path)
            
            # Leer el archivo como GeoDataFrame
            gdf = gpd.read_file(file_path, layer=name_layer[0])
            
            if normalize:
                # Normalizar columnas de texto
                gdf = self._normalize_text_columns(gdf, ["COLONIA", "NOM_MUN", "NOM_ENT"])
                # Remapear nombres de entidades
                gdf = self._remap_entity_names(gdf, "NOM_ENT")
                # Convertir código postal a int64
                gdf = self._convert_postal_code_to_int(gdf, ["CP"])
            
            if as_dataframe:
                return pd.DataFrame(gdf)
            else:
                return gdf
        except (FileNotFoundError, fiona.errors.DriverError) as e:
            print(f"Error al cargar el archivo GPKG: {e}")
            if as_dataframe:
                return pd.DataFrame()
            else:
                return gpd.GeoDataFrame()
```

---
---
```Python
import pandas as pd
import geopandas as gpd
import unicodedata
import fiona


class LoadData:
    """
    Class for loading and preprocessing datasets.
    """
    def __init__(
        self,
        data_dir: str = "../DataSets_Iniciales/coordenadas_obtenidas_google_maps/",
        clean_dir: str = "../DataSets_Iniciales/coordenadas_obtenidas_google_maps/Coordernadas_obtenidas_clasificadas/",
        cp_catalog_dir: str = "../CP_CatalogoNacionaldeCodigosPostales/",
        gpkg_path: str = "../DataSets_Iniciales/colonias_indice_marginacion_cdmx/indice_marginacion_colonias_valle_mexico.gpkg",
    ):
        self.DATA_DIR = data_dir
        self.DATA_DIR_CLEAN = clean_dir
        self.DATA_DIR_EMPTY = f"{clean_dir}nulos/"
        self.DATA_DIR_DIFFERENT = f"{clean_dir}diferentes/"
        self.DATA_DIR_CORRECT = f"{clean_dir}iguales/"
        self.CP_CATALOG_DIR = cp_catalog_dir
        self.GPKG_PATH = gpkg_path

    @staticmethod
    def normalize_text(text:
                       str) -> str:
        """
        Lowercase and remove accents from a text string.
        """
        if pd.isnull(text):
            return text
        text_nfkd = unicodedata.normalize('NFKD', str(text))
        text_ascii = "".join(c for c in text_nfkd if not unicodedata.combining(c))
        return text_ascii.lower()

    def _standardize_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply uniform preprocessing: normalize text, convert postal codes, remap entities.
        """
        df = df.copy()
        for col in df.columns:
            col_low = col.lower()
            # Normalize text for colonia and municipio fields
            if 'col' in col_low or 'colonia' in col_low:
                df[col] = df[col].apply(self.normalize_text)
            if 'mun' in col_low or 'del' in col_low:
                df[col] = df[col].apply(self.normalize_text)
            # Convert postal code columns to Int64
            if 'postal' in col_low or 'cp' in col_low:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
            # Remap entidad names
            if 'ent' in col_low:
                df[col] = df[col].apply(self.normalize_text)
                df[col] = df[col].apply(
                    lambda x: 'cdmx' if isinstance(x, str) and ('cdmx' in x or 'distrito' in x)
                    else 'edomex' if isinstance(x, str) and ('edomex' in x or 'estado' in x or 'mex' in x)
                    else x
                )
        return df

    def load_original_data(self) -> dict:
        """
        Load original CSV files into a dict of DataFrames.
        """
        years = [2020, 2021, 2022, 2023]
        datasets = {
            f"df_coordenadas_sustentantes_{year}":
            pd.read_csv(f"{self.DATA_DIR}coordenadas_sustentantes_{year}.csv", encoding="utf-8")
            for year in years
        }
        return {name: self._standardize_df(df) for name, df in datasets.items()}

    def load_null_classification(self) -> dict:
        """
        Return dataframes where postal_code_api is null.
        """
        orig = self.load_original_data()
        return {name: df[df['postal_code_api'].isna()] for name, df in orig.items()}

    def load_clean_data(self) -> dict:
        """
        Return dataframes without null postal_code_api.
        """
        orig = self.load_original_data()
        clean = {name: df.dropna(subset=['postal_code_api']).copy() for name, df in orig.items()}
        for df in clean.values():
            df['postal_code_api'] = df['postal_code_api'].astype(int)
        return clean

    def load_equal_classification(self) -> dict:
        """
        Return dataframes where postal_code_api == SUS_CP.
        """
        clean = self.load_clean_data()
        return {name: df[df['postal_code_api'] == df['SUS_CP']] for name, df in clean.items()}

    def load_different_classification(self) -> dict:
        """
        Return dataframes where postal_code_api != SUS_CP.
        """
        clean = self.load_clean_data()
        return {name: df[df['postal_code_api'] != df['SUS_CP']] for name, df in clean.items()}

    def load_empty_datasets(self) -> dict:
        """
        Load processed CSVs with missing postal codes (still empty).
        """
        years = [2020, 2021, 2022, 2023]
        datasets = {
            f"df_coordenadas_sustentantes_{year}_nulos":
            pd.read_csv(f"{self.DATA_DIR_EMPTY}df_coordenadas_sustentantes_{year}_nulos.csv", encoding="utf-8")
            for year in years
        }
        return {name: self._standardize_df(df) for name, df in datasets.items()}

    def load_filled_datasets(self) -> dict:
        """
        Load processed CSVs with filled postal codes.
        """
        years = [2020, 2021, 2022, 2023]
        datasets = {
            f"df_coordenadas_sustentantes_{year}_completos":
            pd.read_csv(f"{self.DATA_DIR_EMPTY}df_coordenadas_sustentantes_{year}_completos.csv", encoding="utf-8")
            for year in years
        }
        return {name: self._standardize_df(df) for name, df in datasets.items()}

    def load_geopackage(self) -> gpd.GeoDataFrame:
        """
        Load GeoPackage and return GeoDataFrame preserving geometries.
        """
        layers = fiona.listlayers(self.GPKG_PATH)
        gdf = gpd.read_file(self.GPKG_PATH, layer=layers[0])
        # Standardization of attribute columns (excluding geometry) can be applied separately if needed
        return gdf

```
import json
import os
from typing import List
from urllib.parse import urljoin

import pinecone
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import NewsURLLoader
from langchain_community.vectorstores import Pinecone as pcvs
from langchain_openai import OpenAIEmbeddings
from termcolor import cprint

load_dotenv()


def _get_pinecone() -> pinecone.Pinecone:
    """
    Obtiene un cliente de Pinecone.

    Returns:
        pinecone.Client: El cliente de Pinecone.
    """
    return pinecone.Pinecone(
        api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV")
    )


def _ensure_index_exists() -> None:
    pc = _get_pinecone()
    if os.getenv("PINECONE_INDEX_NAME") not in pc.list_indexes().names():
        try:
            pc.create_index(
                name=os.getenv("PINECONE_INDEX_NAME"),
                metric="cosine",
                dimension=1536,
                spec=pinecone.PodSpec(environment=os.getenv("PINECONE_ENV")),
            )
            cprint(f"Índice creado {os.getenv('PINECONE_INDEX_NAME')}", "yellow")
        except Exception as e:
            cprint(
                f"Error al crear el índice {os.getenv('PINECONE_INDEX_NAME')}: {e}",
                "red",
            )


def _get_or_create_vectorstore(namespace: str) -> pcvs:
    """
    Recupera el almacenamiento de vectores para un espacio de nombres dado. Creando un nuevo espacio de nombres si no existe.

    Args:
        namespace (str): El espacio de nombres del almacenamiento de vectores.

    Returns:
        vectorstore (pcvs): El objeto de almacenamiento de vectores.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    vectorstore = pcvs.from_existing_index(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        namespace=namespace,
    )
    return vectorstore


def get_article_urls(index_urls: List[str]) -> List[str]:
    """
    Extrae las URLs de los artículos de noticias de las páginas índice.

    Args:
        index_urls (List[str]): Las URLs de las páginas índice de noticias.

    Returns:
        List[str]: Una lista de URLs de los artículos de noticias.
    """
    try:
        with open("scraped_urls.json", "r") as f:
            scraped_urls = json.load(f)
    except FileNotFoundError:
        scraped_urls = []

    article_urls = []

    for index_url in index_urls:
        response = requests.get(index_url)
        soup = BeautifulSoup(response.text, "html.parser")

        article_links = soup.select(".uap-port-secc-art-tit a")

        article_urls.extend(
            [
                urljoin(index_url, link.get("href"))
                for link in article_links
                if urljoin(index_url, link.get("href")) not in scraped_urls
            ]
        )

    with open("scraped_urls.json", "w") as f:
        json.dump(scraped_urls + article_urls, f)

    return article_urls


def split_and_store_documents(index_urls: List[str], namespace: str) -> None:
    """
    Divide y almacena los documentos en un almacén de vectores.

    Args:
        index_urls (List[str]): Lista de URLs para obtener el contenido del artículo.
        docs (List[Document]): Lista de documentos para dividir y almacenar.
        namespace (str): Espacio de nombres en el que almacenar los documentos divididos.

    Returns:
        None
    """
    loader = NewsURLLoader(
        urls=get_article_urls(index_urls),
        text_mode=True,
        show_progress_bar=True,
        nlp=True,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=200)
    split_docs = splitter.split_documents(docs)

    _ensure_index_exists()

    try:
        vectorstore = _get_or_create_vectorstore(namespace)
        vectorstore.add_documents(documents=split_docs)

        cprint(
            f"{len(split_docs)} vectores añadidos al espacio de nombres '{namespace}'",
            "green",
        )
    except Exception as e:
        cprint(
            f"Hubo un error al intentar añadir el contenido al almacén de vectores: {e}",
            "red",
        )


def main():
    news_urls = [
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_126__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_127__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_128__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_129__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_130__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_133__1.html",
        "https://www.unap.cl/prontus_unap/site/tax/port/all/taxport_87_132__1.html",
    ]

    split_and_store_documents(index_urls=news_urls, namespace="Noticias")


if __name__ == "__main__":
    main()

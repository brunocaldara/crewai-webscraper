from crewai import Agent, Crew, Process, Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

llm = ChatOpenAI(model="gpt-4o-mini")

site_url = "https://www.sapatariaecostura.com.br/"

agente_extracao = Agent(
    role="Extrator de Informações de Site",
    backstory=f"""Você é um especialista em scraping de sites e coleta de informações relevantes.""",
    goal=f"""Coletar informações detalhadas sobre
      os serviços , produtos e operações da empresa a partir
      do site {site_url}. As informações extraídas serão a
      base para que o agente de soluções de IA avalie os
      dados do site . Identifique o 'core business' ou
      'core product' da empresa . Isso é muito importante
      para realizar uma análise personalizada em outras etapas .""",
    verbose=True,
    memory=True,
    llm=llm,
    tools=[scrape_tool]
)

extrair_informacoes_site = Task(
    description=f"Extrair informações detalhadas sobre os serviços, produtos e operações da empresa a partir do site {
        site_url}.",
    expected_output="Um relatório detalhado contendo as informações sobre serviços, produtos e operações da empresa.",
    tools=[scrape_tool],
    agent=agente_extracao
)

agente_solucoes_ia = Agent(
    role="Pesquisador de Soluções de IA ",
    backstory=f""" Você tem amplo conhecimento em tecnologias de IA
      e é capaz de encontrar as melhores soluções para diferentes tipos de empresas .""",
    goal=f""" Você receber á dados do site {site_url} e o " core
      business " ou " core product " identificado . Baseado na extração,
      você deve encontrar soluções de Inteligência Artificial que
      sejam econômicas e que possam ser integradas nos processos
      da empresa . Solu ções econ ô micas tamb ém são aquelas que
      reduzem o custo da empresa . Leve sempre em consideração
      o " core business " ou " core product " da empresa .""",
    verbose=True,
    memory=True,
    llm=llm,
    tools=[search_tool]
)

pesquisar_solucoes_ia = Task(
    description="""Pesquisar soluções de IA econômicas que podem
      ser integradas nos processos da empresa . Considerar
      ferramentas e tecnologias que sejam
      acessíveis e eficientes.""",
    expected_output="Uma lista de soluções de IA com detalhes sobre custos, benefícios e possíveis aplicações.",
    tools=[search_tool],
    agent=agente_solucoes_ia
)

analista_swot = Agent(
    role="Analista e Consultor SWOT",
    backstory=f""" Você é um especialista em análise estratégica ,
      utilizando a matriz SWOT para fornecer uma avaliação abrangente
      das forças , fraquezas , oportunidades e ameaças de uma
      organização. Este mé todo analítico permite identificar e
      alavancar pontos fortes internos , mitigar fraquezas , explorar
      oportunidades de mercado e prever ameaças externas .
      Ao aplicar a matriz SWOT , você ajuda empresas a desenvolver
      estratégias eficazes que maximizam seu potencial competitivo ,
      aprimoram a tomada de decisões e promovem um crescimento
      sustentável. Sua abordagem detalhada e orientada por dados
      garante que cada análise seja adaptada às necessidades
      específicas do cliente , resultando em planos de ação precisos
      e impactantes.""",
    goal=f""" Analisar as informações coletadas e fornecer análises
      estratégicas detalhadas e precisas para ajudar a empresa a
      identificar e alavancar suas forças , mitigar fraquezas , explorar
      oportunidades de mercado e prever ameaças externas .""",
    verbose=True,
    memory=True,
    llm=llm,
    tools=[search_tool]
)

analise_swot = Task(
    description="""Analisar as informações coletadas e fornecer
      análises estratégicas detalhadas e precisas utilizando a
      matriz SWOT . Identificar forças , fraquezas , oportunidades
      e ameaças da empresa com base nas informações coletadas .""",
    expected_output="""Um relatório detalhado com aanálise SWOT
      da empresa , incluindo forças , fraquezas , oportunidades
      e ameaças.""",
    tools=[search_tool],
    agent=analista_swot
)

analista_financiamento = Agent(
    role="Analista de Financiamento Empresarial",
    backstory=f""" Você é um especialista em financiamento empresarial ,
      encontrando formas inovadoras e eficazes de captar recursos para
      empresas . Com um profundo conhecimento das opções de financiamento
      disponíveis ( privados ou governamentais ), desde empréstimos
      tradicionais até investimentos de capital de risco e crowdfunding ,
      você ajuda as empresas a identificar e acessar as melhores fontes
      de capital para suas necessidades específicas .
      Sua habilidade em negociar termos favoráveis e sua perspicácia
      em elaborar propostas atraentes garantem que os negócios obtenham
      o financiamento necessário para crescer e prosperar .
      Com sua orientação, as empresas conseguem não apenas assegurar
      recursos financeiros , mas também fortalecer sua posição no mercado
      e atingir seus objetivos estratégicos .""",
    goal=f""" Procurar na web por oportunidades reais de financiamento
      da adoção de IA pela empresa . Identificar e alavancar a implantação
      de inteligência artificial na empresa . Não seja genérico nas
      sugestões , quero uma análise personalizada e real , com possíveis
      valores e instituições de financiamento. """,
    verbose=True,
    memory=True,
    llm=llm,
    tools=[search_tool]
)

financiamento_estrategico = Task(
    description=f""" Realizar uma análise detalhada das opções de
      financiamento disponíveis para a empresa . Investigar diversas
      fontes de capital possíveis na web para a empresa financiar a
      adoção de inteligencia artificial em seus processos .
      Analisar os termos , benefícios e riscos associados
      a cada opção e recomendar as melhores estratégias para
      captar recursos .""",
    expected_output=f""" Uma análise detalhada com uma comparação
      das opções de financiamento destacando as melhores estratégias
      para a empresa captar recursos . A análise deve conter listas
      e tabelas comparativas que sirva de base para a escrita do
      relatório final .""",
    agent=analista_financiamento
)


agente_analise_recomendacao = Agent(
    role="Consolidar e Analisar Dados de IA",
    goal=f""" Consolidar todas as informações extraídas ,
      pesquisadas e analisadas pelos agentes anteriores .
      Crie um relatório final detalhado em formato markdown
      sobre como integrar IA de maneira eficiente e econômica
      na empresa em questão.
      Leve sempre em consideração o ' core business '
      ou ' core product ' identificado para gerar uma análise
      personalizada para a empresa . ’Não seja genérico !
      Especialize o texto da análise com base em todas as
      informações disponíveis anteriormente . Gere um texto
      com uma combinação equilibrada de parágrafos contínuos ,
      subtópicos e listas ( quando necessário) e não se
      preocupe com o consumo dos tokens de saída.
      Quero uma análise detalhada . """,
    backstory=f""" Você é um consultor experiente em
      tecnologias de IA e tem um histórico comprovado de
      ajudar empresas a implementar soluções tecnológicas .
      Sua habilidade em consolidar informações de múltiplas
      fontes garante que você possa fornecer uma visão
      abrangente e detalhada .""",
    verbose=True,
    memory=True,
    llm=llm
)

analisar_recomendar = Task(
    description=f""" Gerar um relatório final e consolidar
      todas as informações extraídas , pesquisadas e analisadas
      pelos agentes anteriores . Fornecer recomendações
      detalhadas sobre como integrar IA de maneira econômica
      na empresa . O relatório deve incluir um plano de ação e
      sugestões específicas . """,
    expected_output=f""" Um relatório detalhado com
      recomenda ções sobre :
      1 - Soluções de integração de IA na empresa ;
      2 - Análise Swot ;
      3 - Plano de ação;
      4 - Possibilidades de financiamento ;
      Não seja genérico . Quero números e reais possibilidades .
      5 - Sugestões específicas .
      A saída deve ser em Markdown . É muito importante que
      o texto não contenha coisas como "‘‘‘ markdown " e "‘‘‘". """,
    output_file="analise.md",
    tools=[search_tool],
    agent=agente_analise_recomendacao
)

crew = Crew(
    agents=[agente_extracao,
            agente_solucoes_ia,
            analista_swot,
            analista_financiamento, agente_analise_recomendacao],
    tasks=[extrair_informacoes_site,
           pesquisar_solucoes_ia,
           analise_swot,
           financiamento_estrategico, analisar_recomendar],
    process=Process.sequential
)

result = crew . kickoff(inputs={"site_url": site_url})
print(result)

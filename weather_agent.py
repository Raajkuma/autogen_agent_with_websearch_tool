import asyncio
import os

import streamlit as st
from ddgs import DDGS
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient


# --------------------------------------------------
# Local Qwen model configuration
# --------------------------------------------------

MODEL_NAME = os.getenv(
    "LOCAL_LLM_MODEL",
    "ai/qwen3-vl:2B-BF16",
)

BASE_URL = os.getenv(
    "LOCAL_LLM_BASE_URL",
    "http://localhost:12434/v1",
)


# --------------------------------------------------
# Streamlit page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Weather News AI Agent",
    page_icon="🌦️",
    layout="centered",
)

st.title("🌦️ Weather News AI Agent")

st.write(
    "Get today's weather updates and recent weather news "
    "for any city using an AI-powered assistant."
)

st.caption("Powered by AutoGen, DuckDuckGo and Local Qwen LLM")


# --------------------------------------------------
# DuckDuckGo search tool
# --------------------------------------------------

def search_weather_and_news(city: str) -> str:
    """
    Search the web for today's weather conditions and recent
    weather-related news for a specified city.

    Use this tool whenever the user requests current weather,
    today's forecast, or recent weather news for a city.
    This tool is specifically for weather information, not
    general knowledge or unrelated news.

    Args:
        city: Name of the city to search for.

    Returns:
        Search results containing weather information and news.
    """

    city = city.strip()

    if not city:
        return "Please provide a valid city name."

    try:
        with DDGS() as ddgs:

            # Search for today's weather
            weather_results = list(
                ddgs.text(
                    f"current weather today temperature forecast {city}",
                    max_results=5,
                )
            )

            # Search for recent weather-related news
            news_results = list(
                ddgs.news(
                    f"weather news {city}",
                    timelimit="d",
                    max_results=5,
                )
            )

        output = []

        output.append(f"WEATHER RESULTS FOR {city}")

        for result in weather_results:
            output.append(
                f"""
                Title: {result.get('title', '')}
                Details: {result.get('body', '')}
                Source: {result.get('href', '')}
                """
            )

        output.append(f"RECENT WEATHER NEWS FOR {city}")

        for result in news_results:
            output.append(
                f"""
                Title: {result.get('title', '')}
                Details: {result.get('body', '')}
                Date: {result.get('date', '')}
                Source: {result.get('url', '')}
                """
            )

        if not weather_results and not news_results:
            return f"No weather information found for {city}."

        return "\n".join(output)

    except Exception as exc:
        return (
            f"Search failed: {type(exc).__name__}: {exc}"
        )


# --------------------------------------------------
# AutoGen agent execution
# --------------------------------------------------

async def generate_weather_news(city: str) -> str:

    model_client = OpenAIChatCompletionClient(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key="local-model",
        model_info={
            "vision": True,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
        },
    )

    try:

        weather_agent = AssistantAgent(
            name="weather_news_agent",

            model_client=model_client,

            tools=[
                search_weather_and_news
            ],

            reflect_on_tool_use=True,

            system_message=(
                "You are a professional Weather News Assistant. "

                "Your responsibility is to provide current weather "
                "information and recent weather-related news for "
                "the requested city. "

                "Always call the search_weather_and_news tool "
                "before answering. "

                "Use only information returned by the search tool. "
                "Never invent temperatures, forecasts, news, "
                "or weather conditions. "

                "If reliable current weather information is missing, "
                "clearly mention that it could not be verified. "

                "Your final response MUST contain exactly 3 short lines: "

                "Line 1: Today's weather conditions and temperature "
                "or forecast, if available. "

                "Line 2: One important recent weather-related news update. "

                "Line 3: A practical weather advisory. "

                "Do not include headings, introductions, or additional "
                "paragraphs. Keep every line concise."
            ),
        )

        task = (
            f"Find today's weather conditions and recent weather news "
            f"for {city}. Use your search tool and provide the result "
            f"in exactly 3 short lines."
        )

        result = await weather_agent.run(task=task)

        # Get the final assistant response
        for message in reversed(result.messages):

            if (
                getattr(message, "source", None)
                == "weather_news_agent"
            ):
                content = getattr(message, "content", "")

                if isinstance(content, str) and content.strip():
                    return content.strip()

        return "The agent did not return a valid weather report."

    finally:
        await model_client.close()


# --------------------------------------------------
# Streamlit user interface
# --------------------------------------------------

st.divider()

city = st.text_input(
    "📍 Enter City Name",
    placeholder="e.g. Chennai, London, New York",
)

generate = st.button(
    "🌤️ Get Weather News",
    type="primary",
    use_container_width=True,
)

if generate:

    if not city.strip():

        st.warning("Please enter a city name.")

    else:

        with st.spinner(
            f"Searching weather and news for {city}..."
        ):

            try:

                response = asyncio.run(
                    generate_weather_news(city.strip())
                )

                st.success(
                    f"Weather report generated for {city.title()}!"
                )

                st.subheader(
                    f"🌍 Weather Update: {city.title()}"
                )

                # Display the three-line response
                st.markdown(response)

                st.download_button(
                    label="⬇️ Download Weather Report",
                    data=response,
                    file_name=f"{city.strip().replace(' ', '_')}_weather_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )

            except Exception as exc:

                st.error(
                    "Unable to generate the weather report. "
                    "Please check the local model server and "
                    "its tool-calling compatibility."
                )

                with st.expander("Technical error details"):
                    st.code(
                        f"{type(exc).__name__}: {exc}"
                    )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI Weather News Agent | "
    "Microsoft AutoGen + DuckDuckGo + Local Qwen"
)
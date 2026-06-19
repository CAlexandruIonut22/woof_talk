const PROFILE_KEY = "woof-talk-dog-profile-v1";

function value(id) {
  return document.getElementById(id)?.value?.trim() || "";
}

function checkedValues(selector) {
  return [...document.querySelectorAll(selector)]
    .filter((input) => input.checked)
    .map((input) => input.value);
}

export function getDogProfile() {
  return {
    name: value("dogName"),
    breed: value("dogBreed"),
    age: value("dogAge"),
    baseline: value("dogBaseline")
  };
}

export function getSceneContext() {
  return {
    trigger: value("sceneTrigger"),
    bodyLanguage: checkedValues("[data-body-language]"),
    notes: value("sceneNotes")
  };
}

export function saveDogProfile() {
  const profile = getDogProfile();
  localStorage.setItem(PROFILE_KEY, JSON.stringify(profile));
  return profile;
}

export function hydrateDogProfile() {
  try {
    const saved = JSON.parse(localStorage.getItem(PROFILE_KEY) || "{}");

    const dogName = document.getElementById("dogName");
    const dogBreed = document.getElementById("dogBreed");
    const dogAge = document.getElementById("dogAge");
    const dogBaseline = document.getElementById("dogBaseline");

    if (dogName && saved.name) dogName.value = saved.name;
    if (dogBreed && saved.breed) dogBreed.value = saved.breed;
    if (dogAge && saved.age) dogAge.value = saved.age;
    if (dogBaseline && saved.baseline) dogBaseline.value = saved.baseline;
  } catch {
    // Ignore broken localStorage data.
  }
}